"""Working out who says each line.

Three layers, in order of confidence: word-level alignment against the matched
transcript, neighbour inference across gaps, and voice-embedding clustering
(sherpa-onnx) as the fallback — the only layer available in no-transcript mode.
"""

import json
import re
from difflib import SequenceMatcher

import httpx
import numpy as np
import soundfile as sf

from . import profiles
from .config import SPK_MODEL_PATH, SPK_MODEL_SHA256, SPK_MODEL_URL
from .episodes import canonical_speaker, ensure_scripts, identify_episode, norm_words
from .llm import llm_json_model
from .timing import MIN_GAP
from .transcribe import measure_line_genders
from .translate import CHARS_PER_SEC
from .util import file_key


def _segment_token_times(seg):
    """Per-normalised-token (start,end) times for one Whisper line, so a matched
    word can be attributed to a real spoken time. The cached `words` list carries
    only [start,end] (no text) and drops sub-tokens, so the k-th match-token is
    mapped to a real word timestamp by proportional index (monotonic)
    with no
    word timestamps we fall back to an even split of the segment span."""
    toks = norm_words(seg.get("en", ""))
    m = len(toks)
    if m == 0:
        return []
    words = [w for w in (seg.get("words") or [])
             if isinstance(w, (list, tuple)) and len(w) >= 2 and w[1] > w[0]]
    if words:
        W = len(words)
        res = []
        for t in range(m):
            wi = int(round(t * (W - 1) / (m - 1))) if m > 1 else 0
            wi = max(0, min(W - 1, wi))
            res.append((float(words[wi][0]), float(words[wi][1])))
        return res
    s0 = float(seg.get("start", 0.0))
    e0 = float(seg.get("end", s0 + 0.4))
    span = max(e0 - s0, 0.3)
    return [(s0 + span * t / m, s0 + span * (t + 1) / m) for t in range(m)]


def _word_alignment(whisper_lines, script_lines):
    """The shared monotonic word-level alignment (fuzzy by construction: matches
    on word sequences, so mishearings and dropped words still land right).

    Returns (votes, spans):
      votes -- per Whisper line, {script_line_index: matched_word_count}
               (exactly what align_speakers used to compute internally).
      spans -- per SCRIPT line, [onset, end, n_matched] measured from the Whisper
               WORD timestamps of the words that aligned to it. This is the
               script-first spine: it says WHERE each script line is spoken.
    """
    w_words, w_owner, w_time = [], [], []
    for i, ln in enumerate(whisper_lines):
        toks = norm_words(ln.get("en", ""))
        times = _segment_token_times(ln)
        for t, wd in enumerate(toks):
            w_words.append(wd)
            w_owner.append(i)
            w_time.append(times[t] if t < len(times) else None)
    s_words, s_owner = [], []
    for j, ln in enumerate(script_lines):
        for wd in norm_words(ln.get("text", "")):
            s_words.append(wd)
            s_owner.append(j)
    votes = [{} for _ in whisper_lines]
    spans = {}
    if not w_words or not s_words:
        return votes, spans
    sm = SequenceMatcher(None, w_words, s_words, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            wi = a + k
            i = w_owner[wi]
            j = s_owner[b + k]
            votes[i][j] = votes[i].get(j, 0) + 1
            tm = w_time[wi]
            if tm is None:
                continue
            s, e = tm
            cur = spans.get(j)
            if cur is None:
                spans[j] = [s, e, 1]
            else:
                cur[0] = min(cur[0], s)
                cur[1] = max(cur[1], e)
                cur[2] += 1
    return votes, spans


def _votes_to_idxs(whisper_lines, votes):
    """Per-Whisper-line best script index (or None) -- the original contract."""
    out = []
    for i, v in enumerate(votes):
        if not v:
            out.append(None)
            continue
        j = max(v, key=v.get)
        need = max(len(norm_words(whisper_lines[i].get("en", ""))), 1)
        out.append(j if v[j] / need >= 0.34 else None)
    return out


def align_speakers(whisper_lines, script_lines):
    """Word-level monotonic alignment -> per-Whisper-line script index (or None).

    Same contract as before (used by the casting path)
    it now delegates to the
    shared core that ALSO yields per-script-line spans (see script_line_spans),
    which is what lets a script line Whisper only half-heard still be located.
    """
    votes, _ = _word_alignment(whisper_lines, script_lines)
    return _votes_to_idxs(whisper_lines, votes)


def script_line_spans(whisper_lines, script_lines, min_frac=0.34):
    """Per-SCRIPT-line spoken span from the SAME monotonic matching, for lines
    that are confidently anchored (>= min_frac of their words matched, and either
    >= 2 words matched or the line is short). Returns {j: (onset, end, n)}. Lines
    that are NOT anchored are absent -- the fallback places those between their
    anchored neighbours. `spans` may be passed pre-computed to avoid re-matching."""
    _, spans = _word_alignment(whisper_lines, script_lines)
    return _anchored_spans(spans, script_lines, min_frac)


def _anchored_spans(spans, script_lines, min_frac=0.34):
    out = {}
    for j, ln in enumerate(script_lines):
        sp = spans.get(j)
        if not sp:
            continue
        onset, end, n = sp
        swc = max(len(norm_words(ln.get("text", ""))), 1)
        if n >= 1 and n / swc >= min_frac and (n >= 2 or swc <= 4):
            out[j] = (round(float(onset), 3),
                      round(max(float(end), float(onset) + 0.3), 3), int(n))
    return out


def _fuzzy(a, b):
    wa, wb = set(norm_words(a)), set(norm_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def infer_gaps(whisper_lines, idxs, script_lines):
    """Fill unmatched lines using the SCRIPT itself, not guesswork.
    A Whisper line sits between two matched script positions, so the speaker
    must come from that window: either the previous speaker still talking, or
    whoever answers them. Picks the best fuzzy candidate in the window."""
    n = len(whisper_lines)
    for i in range(n):
        if idxs[i] is not None:
            continue
        prev_j = next((idxs[k] for k in range(i - 1, -1, -1)
                       if idxs[k] is not None), None)
        next_j = next((idxs[k] for k in range(i + 1, n)
                       if idxs[k] is not None), None)
        lo = prev_j if prev_j is not None else -1
        hi = next_j if next_j is not None else len(script_lines)
        window = list(range(max(lo, 0), min(hi + 1, len(script_lines))))
        if not window:
            continue
        text = whisper_lines[i]["en"]
        scored = sorted(((_fuzzy(text, script_lines[j]["text"]), j)
                         for j in window), reverse=True)
        best_sc, best_j = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_sc >= 0.30 and best_sc >= second + 0.10:
            idxs[i] = best_j                      # confident, clearly-best match
        elif prev_j is not None and next_j is not None and next_j - prev_j == 2:
            idxs[i] = prev_j + 1                  # exactly one line missing here
        elif prev_j is not None and next_j is not None and next_j == prev_j:
            idxs[i] = prev_j                      # same script line, split by Whisper
        # else: leave None -> speaker filled from neighbours later (safer than
        # grabbing a wrong script line on a weak match)
    return idxs


MIN_SPINE_COVERAGE = 0.35     # if fewer than this fraction of script lines were


def _resolve_fallback_speaker(raw, neighbor_speaker=None):
    """Speaker label for a missed script line (used for translation context and,
    if ever voiced, casting). Never gates audio -- fallbacks are subtitle-only."""
    c = canonical_speaker(raw)
    if c in profiles.main_cast():
        return c
    if c == "GROUP":
        return neighbor_speaker or "OtherFemale"
    nm = re.sub(r"\(.*?\)", "", raw or "").strip().strip(":").strip().title()
    return nm if 1 < len(nm) <= 30 else "OtherMale"


def insert_missing_script_lines(script, script_lines, spans):
    """Make sure EVERY script line is present exactly once.

    `script` is the list of already-built dubbed lines (each anchored one carries
    its script index in `sidx`). Any script line with no covering dubbed line is
    appended as a SUBTITLE-ONLY fallback with a provisional even-spaced time in
    the empty span between its anchored neighbours (final timing is redone in
    render against the real placed audio). Returns (new_script, n_added).

    HARD INVARIANT: a script line already anchored is never also added as a
    fallback -- no doubling, words never spoken twice.
    """
    import bisect
    spans = spans or {}
    covered = sorted({ln["sidx"] for ln in script
                      if ln.get("sidx") is not None})
    total = len(script_lines)
    if not covered or total == 0 or len(covered) / total < MIN_SPINE_COVERAGE:
        return script, 0                 # nothing anchored / spine not trusted

    # authoritative onset/end per anchored script line: prefer the word-matched
    # span; else the dubbed line's own word window.
    anchor_time = {}
    for j in covered:
        if j in spans:
            anchor_time[j] = (float(spans[j][0]), float(spans[j][1]))
    for ln in script:
        j = ln.get("sidx")
        if j is None or j in anchor_time:
            continue
        s0 = float(ln.get("start", 0.0))
        e0 = float(ln.get("end", s0 + 0.4))
        anchor_time[j] = (s0, max(e0, s0 + 0.3))

    covset = set(covered)
    missing = [j for j in range(total) if j not in covset]
    if not missing:
        return script, 0
    last_end = max(t[1] for t in anchor_time.values()) if anchor_time else 0.0

    added = []

    def flush(grp):
        if not grp:
            return
        pos = bisect.bisect_left(covered, grp[0])
        prev_j = covered[pos - 1] if pos > 0 else None
        next_j = covered[pos] if pos < len(covered) else None
        lo = anchor_time[prev_j][1] if prev_j is not None else 0.0
        hi = anchor_time[next_j][0] if next_j is not None else last_end + 3.0
        if hi <= lo + 1e-6:
            hi = lo + 2.0 * len(grp)          # degenerate: lay out sequentially
        step = (hi - lo) / (len(grp) + 1)
        for k, j in enumerate(grp, start=1):
            txt = script_lines[j].get("text", "")
            spk = _resolve_fallback_speaker(script_lines[j].get("speaker", ""))
            mid = lo + step * k
            est = max(2.0, 0.45 * len(txt.split()))   # generous -> good tx budget
            st = max(lo, mid - est / 2.0)
            en = min(hi, st + est)
            added.append({"start": round(st, 3),
                          "end": round(max(en, st + 0.5), 3),
                          "en": txt, "speaker": spk, "zh": "", "emo": "",
                          "words": [], "sidx": j, "fallback": True})

    grp, prev = [], None
    for j in missing:                         # contiguous missing runs share anchors
        if prev is None or j == prev + 1:
            grp.append(j)
        else:
            flush(grp)
            grp = [j]
        prev = j
    flush(grp)

    if not added:
        return script, 0
    script = script + added
    script.sort(key=lambda ln: (float(ln.get("start", 0.0)),
                                ln.get("sidx") if ln.get("sidx") is not None
                                else 1e9))
    return script, len(added)


def _largest_quiet_subspan(lo, hi, silences):
    """Restrict [lo,hi] to the largest detected silence overlapping it, so a
    fallback subtitle lands in genuinely empty air; falls back to [lo,hi]."""
    best = None
    for s, e in (silences or []):
        a, b = max(lo, s), min(hi, e)
        if b - a > (best[1] - best[0] if best else 0):
            best = (a, b)
    if best and best[1] - best[0] >= 0.4:
        return best
    return (lo, hi)


def _even_layout(items, lo, hi, min_dur, min_gap):
    entries = []
    n = len(items)
    if n == 0:
        return entries
    slot = max(hi - lo, 0.0) / n
    for k, (_, ln) in enumerate(items):
        st = lo + k * slot
        text = (ln.get("zh") or ln.get("en", "") or "").strip()
        want = max(min_dur, min(3.5, len(text) / CHARS_PER_SEC)) if text else min_dur
        en = st + min(want, max(slot - min_gap, min_dur))
        entries.append({"start": round(st, 3), "end": round(max(en, st + min_dur), 3),
                        "zh": text})
    return entries


def place_fallback_subtitles(script, placed_by_sidx, silences, total_len,
                             min_dur=1.0, min_gap=MIN_GAP):
    """Final timing for the subtitle-only fallback lines: distribute them evenly
    in the empty span between the ACTUAL placed audio of their anchored
    neighbours, respecting MIN_GAP and biasing into detected silence. No audio,
    so the no-overlap / tempo / ducking rules are untouched. Returns srt entries."""
    import bisect
    from collections import defaultdict
    fbs = [(i, ln) for i, ln in enumerate(script) if ln.get("fallback")]
    if not fbs:
        return []
    anchored = sorted(placed_by_sidx.keys())
    groups, free = defaultdict(list), []
    for i, ln in fbs:
        j = ln.get("sidx")
        if j is None or not anchored:
            free.append((i, ln))
            continue
        pos = bisect.bisect_left(anchored, j)
        prev_j = anchored[pos - 1] if pos > 0 else None
        next_j = anchored[pos] if pos < len(anchored) else None
        groups[(prev_j, next_j)].append((i, ln))
    entries = []
    for (prev_j, next_j), items in groups.items():
        items.sort(key=lambda t: (t[1].get("sidx")
                                  if t[1].get("sidx") is not None else 0))
        lo = placed_by_sidx[prev_j][1] + min_gap if prev_j is not None else 0.0
        hi = (placed_by_sidx[next_j][0] - min_gap if next_j is not None
              else float(total_len))
        if hi <= lo:
            hi = lo + max(min_dur, 0.6) * len(items)
        qlo, qhi = _largest_quiet_subspan(lo, hi, silences)
        entries.extend(_even_layout(items, qlo, qhi, min_dur, min_gap))
    for i, ln in free:                        # no anchors at all: use provisional
        st = float(ln.get("start", 0.0))
        en = max(float(ln.get("end", st + min_dur)), st + min_dur)
        text = (ln.get("zh") or ln.get("en", "") or "").strip()
        if text:
            entries.append({"start": round(st, 3), "end": round(en, 3), "zh": text})
    return entries


def ensure_speaker_model():
    if SPK_MODEL_PATH.exists() and file_key(SPK_MODEL_PATH) == SPK_MODEL_SHA256:
        return SPK_MODEL_PATH
    SPK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("  downloading the voice-recognition model (29 MB, one time)...")
    tmp = SPK_MODEL_PATH.with_suffix(".part")
    with httpx.Client(timeout=300, trust_env=True, follow_redirects=True) as c:
        with c.stream("GET", SPK_MODEL_URL) as r:
            if r.status_code != 200:
                raise RuntimeError(
                    f"could not download the voice-recognition model "
                    f"(HTTP {r.status_code}). github.com must be reachable "
                    f"for this one-time 29 MB download. See the "
                    f"Troubleshooting section of the README if your network "
                    f"blocks it.")
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    f.write(chunk)
    if file_key(tmp) != SPK_MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Speaker model checksum mismatch; download rejected.")
    tmp.replace(SPK_MODEL_PATH)
    return SPK_MODEL_PATH


def voice_fingerprints(wav16_path, segments, progress=None):
    """Voice-prints from the ORIGINAL audio, in 1.5s windows inside each line.
    Speaker embeddings identify WHO is speaking regardless of pitch, shouting
    or whispering. Returns {line_idx: [vectors]}. Raises on failure."""
    import sherpa_onnx
    model = ensure_speaker_model()
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model),
                                                      num_threads=4)
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    audio, sr = sf.read(str(wav16_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win, hop = 1.5, 0.75
    out = {}
    for i, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        if dur < 0.45:
            continue
        vecs = []
        starts = [seg["start"]]
        if dur > win:
            n = int((dur - win) / hop) + 1
            starts = [seg["start"] + k * hop for k in range(n)]
        for st in starts:
            a = int(st * sr)
            b = int(min(st + win, seg["end"]) * sr)
            piece = audio[a:b]
            if len(piece) < int(0.45 * sr):
                continue
            try:
                stm = ex.create_stream()
                stm.accept_waveform(sr, piece)
                stm.input_finished()
                v = np.array(ex.compute(stm), dtype="float32")
                nrm = float(np.linalg.norm(v))
                if nrm > 0:
                    vecs.append(v / nrm)
            except Exception:
                continue
        if vecs:
            out[i] = vecs
        if progress and i % 40 == 0:
            progress(0.10 + 0.25 * i / max(len(segments), 1),
                     desc=f"Recognising voices... {i}/{len(segments)} lines")
    if len(out) < 8:
        raise RuntimeError("too few analysable lines for voice recognition")
    return out


def _kmeans(X, k, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None, :] - np.stack(C)[None, :, :]) ** 2).sum(-1), axis=1)
        pr = d / (d.sum() or 1.0)
        C.append(X[rng.choice(len(X), p=pr)])
    C = np.stack(C)
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab


def spectral_voices(V, kmax=9, p_percentile=0.90):
    """Standard diarization clustering: centre the embeddings (removes the
    recording's channel signature), refine the affinity, then let the
    eigengap decide HOW MANY voices there are. No magic threshold."""
    n = len(V)
    if n < 4:
        return np.zeros(n, dtype=int), 1
    X = V - V.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    A = (X @ X.T + 1.0) / 2.0
    np.fill_diagonal(A, 0.0)
    thr = np.quantile(A, p_percentile, axis=1, keepdims=True)
    A = np.where(A >= thr, A, A * 0.01)
    A = np.maximum(A, A.T)
    d = A.sum(1) + 1e-9
    Dm = 1.0 / np.sqrt(d)
    L = np.eye(n) - (A * Dm[:, None]) * Dm[None, :]
    w, U = np.linalg.eigh(L)
    w = np.clip(w, 0, None)
    top = min(kmax + 1, n)
    k = int(np.argmax(np.diff(w[:top]))) + 1
    k = max(2 if n >= 8 else 1, min(k, kmax))
    if k <= 1:
        return np.zeros(n, dtype=int), 1
    Y = U[:, :k]
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    lab = _kmeans(Y, k)
    order = np.argsort(-np.bincount(lab, minlength=k))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[x] for x in lab]), k


def cluster_voices(fingerprints):
    """Window-level diarization -> each line gets its majority voice."""
    keys, V = [], []
    for i, vecs in fingerprints.items():
        for v in vecs:
            keys.append(i)
            V.append(v)
    # A scene can have every main character plus guests. Never cap below that:
    # capping by dialogue volume (an earlier bug) forced different actors to
    # share one voice, which is the one error that cannot be recovered from.
    kmax = min(10, max(2, len(V) // 3))
    lab, k = spectral_voices(np.stack(V), kmax=kmax)
    from collections import Counter
    line2c = {}
    for i in fingerprints:
        cs = [int(lab[w]) for w in range(len(keys)) if keys[w] == i]
        if cs:
            line2c[i] = Counter(cs).most_common(1)[0][0]
    return line2c, k


def apply_cluster_majority(labels, line2c):
    """One voice cluster = one speaker, everywhere."""
    from collections import Counter
    for cid in set(line2c.values()):
        idxs = [i for i, c in line2c.items() if c == cid]
        counts = Counter(labels[i] for i in idxs if labels[i] != "Skip")
        if not counts:
            continue
        maj = counts.most_common(1)[0][0]
        for i in idxs:
            if labels[i] != "Skip":
                labels[i] = maj
    return labels


def cluster_genders(line2c, n_clusters, audio_genders):
    """Gender of each VOICE, from the majority of its lines. Aggregating over
    a whole episode is robust to shouting, whispering or a silly voice --
    unlike judging any single line."""
    out = {}
    for cid in range(n_clusters):
        gs = [audio_genders[i] for i, c in line2c.items()
              if c == cid and i < len(audio_genders) and audio_genders[i]]
        m, f = gs.count("MALE"), gs.count("FEMALE")
        out[cid] = "MALE" if m > f else "FEMALE" if f > m else ""
    return out


def cluster_system():
    """Built per call, because the show and its cast come from the profile."""
    prof = profiles.active()
    return (
    f"You are casting an episode of {prof.name}. The audio was analysed: each "
    "VOICE "
    "(V0, V1, ...) is one real person, already separated by voice recognition. "
    "You are told each voice's gender and given sample lines. Decide which of "
    "the named characters each voice is. Use the dialogue: who they talk "
    "about, who answers whom, catchphrases, and job/relationship references. "
    "If two voices are obviously the same person recorded in "
    "different scenes, give them the same name. Never give two "
    "clearly different people the same name. "
    'Respond with JSON only: {"clusters":[{"v":<id>,"speaker":"<name>"}]}.')


def role_gender(role):
    if role == "Skip":
        return ""
    return "MALE" if role in profiles.male_roles() else "FEMALE"


def name_voices(cluster_ids, line2c, segments, cgender, sizes, llm_key, model):
    """Name EVERY voice from the dialogue. Measured pitch is passed only as a
    hint -- on TV audio (laugh track, music) it is unreliable, and letting it
    gate the choice previously dumped whole casts into one guest voice.
    Raises loudly on failure: silently guessing is worse than stopping."""
    if not cluster_ids:
        return {}
    desc = []
    for cid in cluster_ids:
        idxs = sorted(i for i, c in line2c.items() if c == cid)
        step = max(len(idxs) // 16, 1)
        sample = [f'    "{segments[i]["en"]}"' for i in idxs[::step][:16]]
        hint = cgender.get(cid) or "unclear"
        desc.append(f"V{cid}: {sizes.get(cid, len(idxs))} lines, "
                    f"voice sounds {hint.lower()}\n" + "\n".join(sample))
    prof = profiles.active()
    men = [c.name for c in prof.characters
           if c.main and c.gender.upper().startswith("M")]
    women = [c.name for c in prof.characters
             if c.main and c.gender.upper().startswith("F")]
    unknown = [c.name for c in prof.characters
               if c.main and not c.gender]
    cast_line = "The main characters are: "
    cast_line += "; ".join(
        part for part in (
            (", ".join(men) + " (men)") if men else "",
            (", ".join(women) + " (women)") if women else "",
            (", ".join(unknown)) if unknown else "",
        ) if part) + ". "
    user = ("Here are the distinct voices in this episode. Name each one.\n"
            + cast_line +
            "Use OtherMale / OtherFemale "
            "for guests and minor characters, or Skip for a voice that is only "
            "singing the theme song / noise.\n"
            "Use the dialogue itself: who they talk about, who answers whom, "
            "their jobs and relationships.\n\n" + "\n\n".join(desc))
    data = llm_json_model(llm_key, cluster_system(), user, model)  # may raise
    out = {}
    for item in data.get("clusters", []):
        v, sp = item.get("v"), item.get("speaker", "")
        if v in cluster_ids and sp in profiles.labels():
            out[v] = sp
    if not out:
        raise RuntimeError("the model named none of the voices")
    for cid in cluster_ids:
        if cid not in out:
            g = cgender.get(cid)
            out[cid] = "OtherFemale" if g == "FEMALE" else "OtherMale"
    return out


def cast_speakers(segments, llm_key, wav16, workdir, use_scripts=True,
                  progress=None):
    """Who says each line. Priority:
      1. the human-written transcript for this episode  (exact)
      2. voice recognition, to fill any line the transcript did not match
      3. the neighbouring line, as a last resort
    Returns (labels, diagnostics, script_lines_or_None, spans_or_None). The last
    two feed the script-first spine
    they are cached so re-runs restore them
    (and the per-segment sidx/stext) without re-casting."""
    cache = workdir / "speakers8.json"     # bumped for the script-first spine
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        sidx = d.get("sidx")
        stext = d.get("stext")
        if sidx and len(sidx) == len(segments):
            for i in range(len(segments)):
                if sidx[i] is not None:
                    segments[i]["sidx"] = sidx[i]
                if stext and i < len(stext) and stext[i]:
                    segments[i]["stext"] = stext[i]
        sl = d.get("script_lines")
        sp = {int(k): tuple(v) for k, v in (d.get("spans") or {}).items()}
        return d["labels"], d["diag"], sl, (sp or None)
    from collections import Counter
    diag = []
    n = len(segments)
    matched_script_lines = None            # the episode's script (spine), if matched
    matched_spans = None                   # {script_idx: (onset, end, n_matched)}

    # ---- 1. transcript (canonical source of truth)
    # No-transcript mode: a profile with no configured source cannot identify
    # the episode or read speakers off a script, so casting falls back to voice
    # clustering alone. Accuracy is noticeably lower; the UI warns about it.
    script_raw = [None] * n
    if use_scripts and not profiles.active().has_transcripts:
        use_scripts = False
        diag.append(
            "WARNING: this profile has no transcript source, so speakers are "
            "grouped by voice alone. Accuracy is noticeably lower than with "
            "transcripts, and the episode cannot be identified.")
    if use_scripts:
        try:
            if progress:
                progress(0.02, desc="Getting the transcripts...")
            ensure_scripts(lambda f, desc="": progress(0.02 + 0.20 * f, desc=desc)
                           if progress else None)
            if progress:
                progress(0.24, desc="Working out which episode this is...")
            ep, lines, score, runner = identify_episode(segments)
            if score >= 0.12:
                # one monotonic word alignment -> BOTH the per-Whisper mapping
                # (casting, unchanged) AND the per-script-line spans (spine).
                votes, sp_raw = _word_alignment(segments, lines)
                idxs = _votes_to_idxs(segments, votes)
                exact = sum(1 for x in idxs if x is not None)
                idxs = infer_gaps(segments, idxs, lines)
                inferred = sum(1 for x in idxs if x is not None) - exact
                for i, j in enumerate(idxs):
                    segments[i]["sidx"] = j
                    if j is not None:
                        script_raw[i] = lines[j]["speaker"]
                        segments[i]["stext"] = lines[j]["text"]
                matched_script_lines = lines
                matched_spans = _anchored_spans(sp_raw, lines)
                matched_frac = (exact + inferred) / max(n, 1)
                clarity = score / (runner or 0.001)
                diag.append(f"Episode identified: {ep} "
                            f"(match {score:.0%}, next best {runner:.0%}).")
                diag.append(f"Speakers from the transcript: {exact} exact "
                            f"+ {inferred} inferred = "
                            f"{exact + inferred}/{n} lines "
                            f"({matched_frac:.0%}).")
                if clarity < 1.6:
                    diag.append("WARNING: the episode match is not clear-cut "
                                f"({score:.0%} vs {runner:.0%}). If this is the "
                                "wrong episode, the whole cast will be wrong - "
                                "check the title above.")
                if matched_frac < 0.55:
                    diag.append("WARNING: less than half of the dialogue matched "
                                "the script. This file may hold two episodes, a "
                                "different cut, or heavy background noise - "
                                "results past this point are guesswork.")
            else:
                diag.append(f"No transcript matched this file "
                            f"(best {score:.0%}) - using voice recognition only.")
        except Exception as e:  # noqa: BLE001
            diag.append(f"Transcripts unavailable ({str(e)[:90]}) - "
                        f"using voice recognition only.")

    # ---- 2. voice recognition (now only a helper, not the whole answer)
    line2c, k = {}, 0
    audio_genders = [""] * n
    try:
        if progress:
            progress(0.30, desc="Measuring the voices...")
        audio_genders = measure_line_genders(wav16, segments)
        fingerprints = voice_fingerprints(
            wav16, segments,
            (lambda f, desc="": progress(0.32 + 0.28 * f, desc=desc))
            if progress else None)
        line2c, k = cluster_voices(fingerprints)
        diag.append(f"Voice recognition: {len(fingerprints)} lines -> {k} voices.")
    except Exception as e:  # noqa: BLE001
        diag.append(f"Voice recognition unavailable ({str(e)[:80]}).")

    cgender = cluster_genders(line2c, k, audio_genders) if line2c else {}

    def guest_for(i):
        g = cgender.get(line2c.get(i, -1), "") or audio_genders[i]
        return "OtherFemale" if g == "FEMALE" else "OtherMale"

    # Opening theme / title sequence, configured per profile. Peiyin ships no
    # lyrics: `theme_words` is empty unless the user fills it in.
    _theme = profiles.active().theme
    _theme_words = set(_theme.theme_words)

    def looks_like_theme(txt):
        if not _theme_words:
            return False
        return len(set(norm_words(txt)) & _theme_words) >= 3

    # ---- combine: transcript first
    labels = [None] * n
    for i in range(n):
        raw = script_raw[i]
        if _theme.skip_intro_seconds and \
                segments[i]["start"] < _theme.skip_intro_seconds:
            # inside the configured title sequence: mute rather than dub
            labels[i] = "Skip"
            continue
        if not raw:
            # An unmatched line whose words look like the sung theme is muted
            # rather than dubbed. Only fires when the profile lists theme_words.
            if looks_like_theme(segments[i]["en"]):
                labels[i] = "Skip"
            continue
        c = canonical_speaker(raw)
        if c in profiles.main_cast():
            labels[i] = c
        elif c == "GROUP":
            labels[i] = None          # 'All:' -> let the voice/neighbour decide
        else:
            nm = re.sub(r"\(.*?\)", "", raw).strip().strip(":").strip().title()
            labels[i] = nm if 1 < len(nm) <= 30 else guest_for(i)

    # ---- 3. fill gaps with voice clusters: a cluster inherits the character
    # the transcript gave to most of its other lines
    if line2c:
        owner = {}
        for cid in set(line2c.values()):
            names = [labels[i] for i, c in line2c.items()
                     if c == cid and labels[i] in profiles.main_cast()]
            if names:
                owner[cid] = Counter(names).most_common(1)[0][0]
        filled = 0
        for i in range(n):
            if labels[i] is None and i in line2c and line2c[i] in owner:
                labels[i] = owner[line2c[i]]
                filled += 1
        if filled:
            diag.append(f"Voice recognition filled {filled} more lines.")

    # ---- 3b. correct rare slips: a line the transcript tied to a guest/Other,
    # but whose VOICE clearly belongs to a main (that main owns >=80% of the
    # voice cluster, with several lines, and the audio gender matches), is
    # snapped back to the main. A real guest forms their own cluster, so this
    # cannot turn a genuine guest into a lead.
    if line2c:
        cluster_main = {}
        for cid in set(line2c.values()):
            members = [i for i, c in line2c.items() if c == cid]
            main_names = [labels[i] for i in members
                          if labels[i] in profiles.main_cast()]
            if not main_names:
                continue
            top, cnt = Counter(main_names).most_common(1)[0]
            if cnt >= 4 and cnt / max(len(members), 1) >= 0.8:
                cluster_main[cid] = top
        corrected = 0
        for i in range(n):
            cid = line2c.get(i)
            if (cid in cluster_main and labels[i] != "Skip"
                    and labels[i] not in profiles.main_cast()):
                m = cluster_main[cid]
                if not audio_genders[i] or audio_genders[i] == role_gender(m):
                    labels[i] = m
                    corrected += 1
        if corrected:
            diag.append(f"Voice recognition corrected {corrected} line(s) back "
                        f"to a main character.")

    # ---- 4. last resort: neighbours, then guest
    still = 0
    for i in range(n):
        if labels[i] is not None:
            continue
        prev = next((labels[j] for j in range(i - 1, -1, -1) if labels[j]), None)
        nxt = next((labels[j] for j in range(i + 1, n) if labels[j]), None)
        gap_p = segments[i]["start"] - segments[i - 1]["end"] if i else 9e9
        gap_n = segments[i + 1]["start"] - segments[i]["end"] if i + 1 < n else 9e9
        labels[i] = (prev if (prev and gap_p <= gap_n) else (nxt or prev
                     or guest_for(i)))
        still += 1
    if still:
        diag.append(f"{still} lines had no match and follow the line next to them.")

    counts = Counter(labels)
    diag.append("Cast: " + ", ".join(f"{sp} ({c} lines)"
                                     for sp, c in counts.most_common()))
    mains = [c for c in counts if c in profiles.main_cast()]
    if len(mains) < 2:
        diag.append("WARNING: almost no main characters were identified. "
                    "Send this Status box to Claude.")
    print("  " + "\n  ".join(diag))
    cache.write_text(json.dumps({
        "labels": labels, "diag": diag,
        "sidx": [segments[i].get("sidx") for i in range(n)],
        "stext": [segments[i].get("stext") for i in range(n)],
        "script_lines": matched_script_lines,
        "spans": {str(j): list(v) for j, v in (matched_spans or {}).items()},
    }, ensure_ascii=False), encoding="utf-8")
    return labels, diag, matched_script_lines, (matched_spans or None)
