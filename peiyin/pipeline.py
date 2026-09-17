"""Stage orchestration and the per-episode cache.

Runs transcribe -> identify -> cast -> translate -> synthesize -> plan -> mix,
caching each stage so a re-run only pays for what actually changed. Importable
and runnable without Gradio, which is what makes the pipeline testable.
"""

import json
from pathlib import Path

from .cache import script_workdir
from .config import DIAG, SR, WORK_ROOT, current_out_dir, llm_key, load_config
from .mix import _fit_key, apply_tempo, assemble_ducked, mux, synth_raw
from .profiles import main_cast, male_roles, speakers
from .speakers import (
    _fuzzy,
    cast_speakers,
    insert_missing_script_lines,
    place_fallback_subtitles,
)
from .timing import SPLIT_GAP, build_line_windows, plan_timeline, silence_from_regions
from .transcribe import detect_speech_regions, transcribe
from .translate import translate_llm
from .tts import (
    build_fish_voice_map,
    ensure_fish_auto_pool,
    ensure_guest_voices,
    get_provider,
)
from .util import file_key, run_ffmpeg, wav_duration, write_srt


def merge_by_script_line(segments, labels):
    """Whisper splits one scripted line into fragments; each fragment used to
    get the FULL script line, so it was spoken 2-3 times and timing collapsed.
    Fragments carrying the same script index (sidx) AND speaker become ONE
    dubbed line -- even if a garbled middle fragment lost its sidx, as long as
    it is bracketed by the same script line on both sides.

    BUT fragments of one script line that are separated by a real PAUSE
    (a breath, a beat, a scene cut) are kept SEPARATE, so a long speech is
    dubbed as several lines with breaks between them -- each hooked to when it
    is actually spoken -- instead of one early-firing blob. A split piece uses
    the words actually heard in it (never the whole script line), so nothing is
    ever spoken twice.
    """
    n = len(segments)
    out_segs, out_labels, merged = [], [], 0
    i = 0
    while i < n:
        sidx = segments[i].get("sidx")
        j = i + 1
        if sidx is not None:
            while j < n:
                sj = segments[j].get("sidx")
                gap = segments[j]["start"] - segments[j - 1]["end"]
                if sj == sidx and labels[j] == labels[i] and gap <= SPLIT_GAP:
                    j += 1
                    continue
                # bridge a garbled middle: same script line resumes just ahead,
                # same speaker, and the gap is short
                if (sj is None and labels[j] == labels[i]
                        and j + 1 < n
                        and segments[j + 1].get("sidx") == sidx
                        and labels[j + 1] == labels[i]
                        and gap <= 1.0
                        and segments[j + 1]["start"] - segments[j]["end"] <= 1.0):
                    j += 1
                    continue
                break
        seg = dict(segments[i])
        seg["end"] = segments[j - 1]["end"]
        # gather every fragment's word timestamps so the merged line keeps a real,
        # ordered word window spanning the whole run.
        allw = []
        for x in segments[i:j]:
            allw += (x.get("words") or [])
        seg["words"] = allw
        # is this run only PART of its script line (i.e. we split it on a pause)?
        prev_same = (sidx is not None and bool(out_segs)
                     and out_segs[-1].get("sidx") == sidx)
        next_same = (sidx is not None and j < n
                     and segments[j].get("sidx") == sidx
                     and labels[j] == labels[i])
        partial = prev_same or next_same
        if sidx is not None and segments[i].get("stext") and not partial:
            heard = " ".join(x["en"] for x in segments[i:j])
            seg["en"] = (segments[i]["stext"]
                         if _fuzzy(heard, segments[i]["stext"]) >= 0.4 else heard)
        else:
            seg["en"] = " ".join(x["en"] for x in segments[i:j])
        seg.pop("stext", None)
        out_segs.append(seg)
        out_labels.append(labels[i])
        if j - i > 1:
            merged += j - i - 1
        i = j
    return out_segs, out_labels, merged


def _reuse_translations(script, workdir, engine):
    """Copy Chinese from an earlier script cache (same engine, same episode) onto
    the SAME lines, so a re-run only pays to translate genuinely new lines.

    Keyed by script-line index first. Keying on the English alone was wrong:
    Dialogue repeats short lines constantly ("What?", "Hi", "Oh my God"), so every
    occurrence inherited the FIRST one's wording AND emotion cue -- which flattens
    the delivery across the whole episode. An English-only match is therefore used
    only when that line has exactly ONE translation in the old cache.
    """
    try:
        by_line, by_en = {}, {}
        for f in sorted(workdir.glob(f"script*_{engine}.json")):
            try:
                old = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for ln in old:
                en = (ln.get("en") or "").strip()
                zh = (ln.get("zh") or "").strip()
                if not en or not zh:
                    continue
                val = (zh, ln.get("emo", ""))
                sidx = ln.get("sidx")
                if sidx is not None:
                    by_line.setdefault((sidx, en), val)
                by_en.setdefault(en, set()).add(val)
        if not by_line and not by_en:
            return
        hits = 0
        for ln in script:
            if ln.get("speaker") == "Skip" or ln.get("zh"):
                continue
            en = (ln.get("en") or "").strip()
            got = by_line.get((ln.get("sidx"), en))
            if got is None:
                cands = by_en.get(en)
                if cands and len(cands) == 1:      # unambiguous -> safe to reuse
                    got = next(iter(cands))
            if got:
                ln["zh"], ln["emo"] = got[0], got[1]
                hits += 1
        if hits:
            DIAG.append(f"Reused {hits} translation(s) from the previous run; "
                        f"only new lines are sent to the translator.")
    except Exception:  # noqa: BLE001
        pass


def build_script(video, llm_key, model_size, workdir,
                 use_scripts=True, progress=None):
    """Stages 1+2+4 -> a full script [{start,end,en,speaker,zh}]. Cached."""
    def report(f, d=""):
        if progress:
            progress(f, desc=d)
    audio_workdir = Path(workdir)
    workdir = script_workdir(audio_workdir, model_size, llm_key, use_scripts)
    workdir.mkdir(parents=True, exist_ok=True)
    engine = "llm"
    script_cache = workdir / f"script9_{engine}.json"    # bumped for script-first
    edited = workdir / "script_edited.json"
    if edited.exists():
        return json.loads(edited.read_text(encoding="utf-8")), True
    if script_cache.exists():
        return json.loads(script_cache.read_text(encoding="utf-8")), False

    report(0.02, "Reading the video file...")
    wav16, _ = extract_audio(video, audio_workdir)
    report(0.05, "Listening to the English (first ever run downloads a model)...")
    segments = transcribe(wav16, model_size, audio_workdir,
                          progress=lambda f, desc="": report(0.05 + 0.45 * f, desc))
    if not segments:
        raise RuntimeError("No speech found in this file.")

    report(0.52, "Working out who says what...")
    labels, cast_diag, script_lines, spans = cast_speakers(
        segments, llm_key, wav16, workdir, use_scripts,
        progress=lambda f, desc="": report(0.52 + 0.18 * f, desc))
    DIAG.extend(cast_diag)
    segments, labels, merged = merge_by_script_line(segments, labels)
    if merged:
        DIAG.append(f"Joined {merged} split fragments back into whole lines "
                    f"({len(segments)} lines to dub).")
    script = [dict(seg, speaker=labels[i], zh="", emo="")
              for i, seg in enumerate(segments)]

    # SCRIPT-FIRST: walk the whole matched script and add any line Whisper missed
    # as a subtitle-only fallback, so no scripted line can disappear.
    if script_lines:
        script, added = insert_missing_script_lines(script, script_lines, spans)
        if added:
            DIAG.append(f"Script-first: recovered {added} script line(s) Whisper "
                        f"never heard, placed as subtitles in the gaps so they "
                        f"aren't dropped ({len(script)} lines total).")

    # Reuse translations from any prior cache so a re-run only translates the new
    # (fallback) lines -- keeps re-running cheap, no API cost for unchanged lines.
    _reuse_translations(script, workdir, engine)

    report(0.72, "Translating...")
    script = translate_llm(script, llm_key, workdir,
                           progress=lambda f, desc="": report(0.72 + 0.25 * f, desc))
    script_cache.write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    return script, False


def extract_audio(video, workdir):
    wav16 = workdir / "audio16k.wav"
    wav_full = workdir / "audio_full.wav"
    if not wav16.exists():
        run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                    str(wav16)], "audio extraction")
    if not wav_full.exists():
        run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", str(SR),
                    str(wav_full)], "audio extraction")
    return wav16, wav_full


def render_from_script(video_path, script, bg_hi, burn_subs,
                       progress=None, fish_key="", use_emotion=True,
                       provider=None):
    provider = provider or get_provider()
    tag = provider.name

    def report(f, d=""):
        if progress:
            progress(f, desc=d)
    video = Path(video_path)
    workdir = WORK_ROOT / file_key(video)
    workdir.mkdir(parents=True, exist_ok=True)
    wav16, wav_full = extract_audio(video, workdir)
    total_len = wav_duration(wav_full)

    report(0.02, "Checking your Chinese voices...")
    # one-time: pull popular public voices so guests never take a cast voice
    ensure_fish_auto_pool(fish_key)
    voice_map = build_fish_voice_map()
    # subtitle-only fallbacks are never voiced, so they need no voice assignment
    guests = {ln["speaker"] for ln in script
              if not ln.get("fallback")
              and ln["speaker"] not in speakers() and ln["speaker"] != "Skip"}
    if guests:
        voice_map.update(ensure_guest_voices(
            guests, llm_key()))
    for ln in script:                      # safety net: never crash on a name
        if (not ln.get("fallback") and ln["speaker"] not in voice_map
                and ln["speaker"] != "Skip"):
            ln["speaker"] = "OtherMale"
    # Hard guarantee: a guest is NEVER given a cast character's voice. If no
    # guest voice could be resolved, stop with a clear message instead.
    # (Fallback lines are subtitle-only, so a missing voice is not an error.)
    missing = sorted({ln["speaker"] for ln in script
                      if not ln.get("fallback") and ln["speaker"] != "Skip"
                      and not voice_map.get(ln["speaker"])})
    if missing:
        raise RuntimeError(
            "No voice is available for: " + ", ".join(missing[:6])
            + ". Cast the main characters in the Cast tab, and add guest "
            "voices (or enable the public-voice pool) so guests have "
            "somewhere to draw from. Guests are never given a cast "
            "character's voice, so nothing was dubbed.")
    print("  voices:", json.dumps(voice_map, ensure_ascii=False))

    report(0.05, "Recording the Chinese voices...")
    # Backup voice per speaker (fish only): a gender-matched generic guest voice,
    # so a line whose own voice fails is recovered instead of vanishing.
    guest_gender = load_config().get("guest_genders", {})

    def _fallback_for(spk):
        if spk in ("OtherMale", "OtherFemale"):
            return ""
        fem = ((spk in main_cast() and spk not in male_roles())
               or guest_gender.get(spk) == "FEMALE")
        return voice_map.get("OtherFemale" if fem else "OtherMale", "")

    fallbacks = {s: _fallback_for(s) for s in {ln["speaker"] for ln in script}}
    raw = synth_raw(script, voice_map, workdir,
                    progress=lambda f, desc="": report(0.05 + 0.45 * f, desc),
                    fish_key=fish_key, use_emotion=use_emotion,
                    fallbacks=fallbacks, provider=provider)
    durations = {i: wav_duration(p) for i, p in raw.items()}

    report(0.50, "Finding the real speech and the silent gaps (neural VAD)...")
    reg_cache = workdir / "speech_regions_v4.json"
    if reg_cache.exists():
        regions = [tuple(x) for x in json.loads(reg_cache.read_text(encoding="utf-8"))]
        source = "cache"
    else:
        regions, source = detect_speech_regions(wav16)
        reg_cache.write_text(json.dumps(regions), encoding="utf-8")
    silences = silence_from_regions(regions, total_len)
    windows = build_line_windows(script, silences, total_len, speech=regions)
    DIAG.append(f"Alignment: {len(regions)} speech region(s) via {source} VAD. "
                f"Each line starts on its onset and plays at normal speed into the "
                f"empty time after it (silences not preserved); it only speeds up "
                f"if the next line's audio leaves no room.")

    report(0.52, "Planning the timeline (play into the gaps, speed up only if needed)...")
    plan = plan_timeline(script, durations, total_len,
                         silences=silences, windows=windows)

    report(0.55, "Fitting the lines...")
    fit_dir = workdir / f"fit2_{tag}"
    fit_dir.mkdir(exist_ok=True)
    placements, entries = [], []
    placed = set()
    placed_by_sidx = {}                       # anchored script line -> placed (s,e)
    for i, place, tempo in plan:
        dst = fit_dir / f"{_fit_key(raw[i], tempo)}.wav"
        if not dst.exists():
            apply_tempo(raw[i], tempo, dst)
        d = durations[i] / tempo
        placements.append((place, dst))
        entries.append({"start": place, "end": place + d, "zh": script[i]["zh"]})
        placed.add(i)
        sidx = script[i].get("sidx")
        if sidx is not None:
            a, b = placed_by_sidx.get(sidx, (place, place + d))
            placed_by_sidx[sidx] = (min(a, place), max(b, place + d))

    # SCRIPT-FIRST fallbacks: script lines Whisper never heard get their Chinese
    # subtitle placed evenly in the empty span between their anchored neighbours'
    # ACTUAL placed audio (respecting MIN_GAP + detected silences). No audio, so
    # the no-overlap / tempo / ducking guarantees are untouched.
    fb_entries = place_fallback_subtitles(script, placed_by_sidx, silences, total_len)
    if fb_entries:
        entries.extend(fb_entries)
        DIAG.append(f"{len(fb_entries)} recovered line(s) shown as subtitles in "
                    f"the gaps (script-first).")

    # Subtitle retention: any line that has text but could NOT be voiced (a voice
    # that failed even on the backup, OR a line that never got translated) still
    # appears in the subtitles at its original time, so a line is never silently
    # lost. Untranslated lines fall back to their original English. (Fallbacks are
    # already handled above and are skipped here so nothing is doubled.)
    lost = 0
    for i, ln in enumerate(script):
        if ln["speaker"] == "Skip" or i in placed or ln.get("fallback"):
            continue
        text = ln.get("zh") or (ln.get("en", "") or "").strip()
        if not text:
            continue
        st = float(ln.get("start", 0.0))
        en = max(float(ln.get("end", st + 1.5)), st + 1.0)
        entries.append({"start": st, "end": en, "zh": text})
        lost += 1
    if lost:
        DIAG.append(f"{lost} line(s) could not be voiced but were kept in the "
                    f"subtitles so nothing is lost.")
    entries.sort(key=lambda e: e["start"])

    report(0.62, "Mixing (laugh track ducks under the Chinese)...")
    dub_wav = workdir / f"dub2_{tag}.wav"
    assemble_ducked(wav_full, placements, bg_hi, dub_wav)

    stem = video.stem + "-" + file_key(video)[:12]
    out_dir = current_out_dir()
    # Burn from a space-free work path (the output folder name may contain
    # spaces, which can break the ffmpeg subtitles filter), then save a copy of
    # the .srt next to the video for the user.
    srt_work = workdir / f"{stem}.chinese.srt"
    write_srt(entries, srt_work)   # subtitle times match the PLACED audio
    srt_path = out_dir / f"{stem}.chinese.srt"
    try:
        srt_path.write_text(srt_work.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        srt_path = srt_work

    report(0.70, "Exporting the final video "
                 + ("(burning subtitles takes a few minutes)..." if burn_subs else "..."))
    out_mp4 = out_dir / f"{stem}.chinese.{tag}.mp4"
    mux(video, dub_wav, srt_work, out_mp4, burn=burn_subs)
    report(1.0, "Done!")
    return str(out_mp4), str(srt_path)
