"""Speech-to-text and speech/silence detection.

faster-whisper with word-level timestamps, hallucination filtering and
merged-line splitting
plus the Silero-VAD speech-region detection and energy
fallback that produce the silence map. These read audio files
the pure
planning maths that consumes their output lives in `peiyin.timing`.
"""

import json
import re

import numpy as np
import soundfile as sf

from .config import DIAG
from .timing import MIN_SILENCE, SIL_CLOSE, SIL_MIN_SPEECH, SIL_THRESH_FRAC
from .util import wav_duration

DASH_SPLIT = re.compile(r"(?<=[.?!…\"])\s+[-–—]\s+")


def split_merged(seg):
    """Whisper sometimes merges two speakers: '- Hey. - What's up?'.
    Split conservatively
    share the time proportionally. Word-level timestamps
    (if present) are sliced to the parts so each half keeps its real timing."""
    text = seg["en"].strip()
    text = re.sub(r"^[-–—]\s*", "", text)
    parts = [p.strip() for p in DASH_SPLIT.split(text) if p.strip()]
    words = list(seg.get("words") or [])
    if len(parts) < 2:
        return [dict(seg, en=text, words=words)]
    total_chars = sum(len(p) for p in parts) or 1
    dur = seg["end"] - seg["start"]
    out, t, wi = [], seg["start"], 0
    for pi, p in enumerate(parts):
        d = dur * len(p) / total_chars
        s0, e0 = round(t, 3), round(t + d, 3)
        # give this part a proportional slice of the words; the last part gets
        # whatever is left so no word is dropped.
        if words:
            if pi == len(parts) - 1:
                pw = words[wi:]
            else:
                take = int(round(len(words) * len(p) / total_chars))
                pw = words[wi:wi + take]
                wi += take
        else:
            pw = []
        if pw:
            s0 = float(pw[0][0])
            e0 = max(float(pw[-1][1]), s0 + 0.3)
        out.append({"start": s0, "end": e0, "en": p, "words": pw})
        t += d
    return out


def transcribe(wav16, model_size, workdir, progress=None):
    # v2 cache (word-level). Named apart from the old segment-level cache so a
    # stale transcript can never feed the new aligner.
    cache = workdir / f"transcript_v2_{model_size}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # word_timestamps=True is the heart of v2: it gives a real start/end for
    # every word, so a line's window is where its words actually are.
    seg_iter, info = model.transcribe(str(wav16), vad_filter=True,
                                      word_timestamps=True)
    total = info.duration or 1.0
    segments, prev_text, repeats = [], None, 0
    for seg in seg_iter:
        t = seg.text.strip()
        if not t:
            continue
        # hallucination filters
        if getattr(seg, "no_speech_prob", 0) > 0.6:
            continue
        if (getattr(seg, "compression_ratio", 0) > 2.4
                and getattr(seg, "avg_logprob", 0) < -0.5):
            continue
        if t == prev_text:
            repeats += 1
            if repeats >= 2:      # 3rd identical line in a row -> hallucination
                continue
        else:
            repeats = 0
        prev_text = t
        # collect this segment's per-word times as [start, end] pairs
        words = []
        for w in (getattr(seg, "words", None) or []):
            ws, we = getattr(w, "start", None), getattr(w, "end", None)
            if ws is None or we is None or we <= ws:
                continue
            words.append([round(float(ws), 3), round(float(we), 3)])
        for piece in split_merged({"start": round(seg.start, 3),
                                   "end": round(seg.end, 3), "en": t,
                                   "words": words}):
            if piece["en"]:
                segments.append(piece)
        if progress:
            progress(min(seg.end / total, 1.0),
                     desc=f"Listening... {int(seg.end)}s / {int(total)}s")
    # enforce monotone, non-overlapping source times (words are left untouched --
    # the aligner reads them directly for the true speech window).
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i - 1]["end"]:
            segments[i]["start"] = segments[i - 1]["end"]
            segments[i]["end"] = max(segments[i]["end"],
                                     segments[i]["start"] + 0.3)
    segments = merge_utterances(segments)
    cache.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return segments


def _frame_f0s(piece, sr, fmin=70.0, fmax=350.0, frame_s=0.040, hop_s=0.020,
               energy_frac=0.05, periodicity_frac=0.45):
    """Per-frame fundamental (f0, Hz) estimates for the VOICED frames of a mono
    signal, by autocorrelation. Shared by the per-line gender measurement and the
    guest-voice pitch probe so both use the SAME estimator (defaults reproduce the
    original gender logic exactly). Returns a list of f0 values."""
    piece = np.asarray(piece, dtype="float32")
    frame = int(frame_s * sr)
    hop = int(hop_s * sr)
    if frame < 8 or hop < 1 or len(piece) < frame:
        return []
    lag_lo = max(int(sr / fmax), 1)            # fmax Hz upper bound
    lag_hi = max(int(sr / fmin), lag_lo + 1)   # fmin Hz lower bound
    emax = float(np.max(piece ** 2)) or 1.0
    f0s = []
    for off in range(0, len(piece) - frame, hop):
        w = piece[off:off + frame]
        w = w - w.mean()
        e0 = float(np.dot(w, w))
        if e0 < energy_frac * emax * frame:    # too quiet -> unvoiced
            continue
        ac = np.correlate(w, w, mode="full")[frame - 1:]
        band = ac[lag_lo:lag_hi]
        if not len(band):
            continue
        k = int(np.argmax(band)) + lag_lo
        if ac[k] < periodicity_frac * ac[0]:   # weak periodicity -> unvoiced
            continue
        f0s.append(sr / k)
    return f0s


def measure_line_genders(wav16_path, segments):
    """Median voice pitch per line from the ORIGINAL audio.
    Returns a list of 'MALE' / 'FEMALE' / '' (unknown). Pure numpy."""
    audio, sr = sf.read(str(wav16_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    out = []
    for seg in segments:
        a = int(max(seg["start"] - 0.03, 0) * sr)
        b = int(min(seg["end"] + 0.03, len(audio) / sr) * sr)
        f0s = _frame_f0s(audio[a:b], sr)       # fmax=350 -> same as before
        if len(f0s) >= 5:
            f0 = float(np.median(f0s))
            out.append("MALE" if f0 <= 155 else "FEMALE" if f0 >= 175 else "")
        else:
            out.append("")
    return out


def estimate_median_f0(wav_path):
    """Median f0 (Hz) over a whole wav, used to screen guest voices by pitch.
    Widens the band to 500 Hz so genuinely shrill voices are measured (not
    folded to a subharmonic). None if it can't be measured."""
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        return None
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    f0s = _frame_f0s(audio, sr, fmax=500.0)
    if len(f0s) >= 5:
        return float(np.median(f0s))
    return None


SENT_END = re.compile(r"[.?!…\"\u2019]$")


def merge_utterances(segments, max_gap=0.25, max_len=8.0):
    """Fragments of one utterance become ONE line, so a single sentence can
    never be split between two speakers. Merge when the previous fragment has
    no sentence-final punctuation and the next starts almost immediately."""
    if not segments:
        return segments
    out = [dict(segments[0])]
    for seg in segments[1:]:
        prev = out[-1]
        gap = seg["start"] - prev["end"]
        joined_len = seg["end"] - prev["start"]
        if (gap <= max_gap and joined_len <= max_len
                and not SENT_END.search(prev["en"].rstrip())):
            prev["en"] = (prev["en"].rstrip() + " " + seg["en"].lstrip()).strip()
            prev["end"] = seg["end"]
            prev["words"] = (prev.get("words") or []) + (seg.get("words") or [])
        else:
            out.append(dict(seg))
    return out


def _runs_shorter_than(mask, k, value):
    """Flip every run equal to `value` that is shorter than k frames.
    Used to close short quiet dips and to drop short speech blips."""
    if k <= 1:
        return mask
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i] == value:
            j = i
            while j < n and m[j] == value:
                j += 1
            if j - i < k:
                m[i:j] = not value
            i = j
        else:
            i += 1
    return m


def speech_silence_map(wav16_path, min_sil=MIN_SILENCE, close_gap=SIL_CLOSE,
                       min_speech=SIL_MIN_SPEECH, thr_frac=SIL_THRESH_FRAC):
    """Find the SILENT gaps in the episode from the 16 kHz audio (energy VAD).

    Returns a sorted list of (start, end) seconds where nobody is talking --
    scene changes, beats between scenes, dead air. Deliberately conservative:
    a stretch is only a gap if it is clearly quiet *and* lasts >= min_sil, so
    ordinary speech is never mistaken for silence. Laughter/music read as
    'speech' (not silence), which is the safe direction here -- we only need the
    genuinely quiet gaps so the planner can keep the Chinese out of them.
    """
    audio, sr = sf.read(str(wav16_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return []
    hop = max(int(0.020 * sr), 1)          # 20 ms frames
    n_hops = audio.shape[0] // hop
    if n_hops < 3:
        return []
    frames = audio[:n_hops * hop].reshape(n_hops, hop)
    power = np.mean(frames.astype(np.float64) ** 2, axis=1)
    # smooth power over ~40 ms (2 hops) so single-sample spikes don't matter
    if n_hops >= 2:
        power = np.convolve(power, np.ones(2) / 2.0, mode="same")
    db = 10.0 * np.log10(power + 1e-12)
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 95))
    if peak - floor < 1e-6:                 # flat/near-silent file: nothing useful
        return []
    thr = floor + thr_frac * (peak - floor)
    speech = db > thr
    frame_s = hop / sr
    speech = _runs_shorter_than(speech, int(round(close_gap / frame_s)), False)
    speech = _runs_shorter_than(speech, int(round(min_speech / frame_s)), True)
    sils, i, n = [], 0, len(speech)
    while i < n:
        if not speech[i]:
            j = i
            while j < n and not speech[j]:
                j += 1
            s = i * frame_s
            e = min(j * frame_s, audio.shape[0] / sr)
            if e - s >= min_sil:
                sils.append((round(s, 3), round(e, 3)))
            i = j
        else:
            i += 1
    return sils


def _speech_from_energy(wav16_path):
    """Fallback speech regions: the complement of the energy silence map."""
    sils = speech_silence_map(wav16_path)
    try:
        dur = wav_duration(wav16_path)
    except Exception:
        info = sf.info(str(wav16_path))
        dur = info.frames / info.samplerate
    regions, t = [], 0.0
    for s, e in sils:
        if s - t > 0.05:
            regions.append((round(t, 3), round(s, 3)))
        t = max(t, e)
    if dur - t > 0.05:
        regions.append((round(t, 3), round(dur, 3)))
    return regions


def detect_speech_regions(wav16_path):
    """Where speech actually happens, as (start, end) seconds.

    Prefers faster-whisper's bundled Silero VAD -- a real speech/non-speech
    model that ships inside the package (no Hugging Face download) and, unlike
    energy thresholding, does NOT fire on the laugh track or incidental music.
    Falls back to the energy map's complement if the neural VAD is unavailable
    or returns too little to be trusted. Returns (regions, source_name).
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        audio, sr = sf.read(str(wav16_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        ts = get_speech_timestamps(
            audio.astype("float32"),
            vad_options=VadOptions(
                min_silence_duration_ms=int(MIN_SILENCE * 1000),
                speech_pad_ms=0),
            sampling_rate=int(sr))
        regions = [(round(d["start"] / sr, 3), round(d["end"] / sr, 3))
                   for d in ts if d.get("end", 0) > d.get("start", 0)]
        if len(regions) >= 3:            # trust it only if it found real structure
            return regions, "neural"
    except Exception as e:  # noqa: BLE001
        DIAG.append(f"Neural VAD unavailable ({str(e)[:60]}); using energy VAD.")
    return _speech_from_energy(wav16_path), "energy"
