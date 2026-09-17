"""Audio assembly, subtitles and muxing.

Synthesis fan-out, tempo fitting, ducking the original dialogue under the dub
while keeping the background bed, SRT writing, CJK font selection and the final
ffmpeg mux with optional burned-in subtitles.
"""

import concurrent.futures
import hashlib
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import DIAG, fish_model
from .tts import get_provider
from .util import run_ffmpeg, to_std_wav

BG_LOW = 0.06           # original audio under Chinese speech
RAMP_S = 0.06           # duck ramp


def synth_raw(script, voice_map, workdir, progress=None,
              fish_key="", use_emotion=True, fallbacks=None, provider=None):
    """Generate every clip at natural speed. Returns idx -> raw wav path.
    If a line's own voice fails (common with flaky public guest voices), it is
    retried with a gender-matched backup voice so the line is not lost."""
    fallbacks = fallbacks or {}
    provider = provider or get_provider()
    pname = provider.name
    model = fish_model() if pname == "fish" else ""
    cache_provider = pname + "|" + model
    tts_dir = workdir / f"tts2_{pname}"
    tts_dir.mkdir(exist_ok=True)
    todo = [i for i, ln in enumerate(script)
            if ln["speaker"] != "Skip" and ln.get("zh")
            and not ln.get("fallback")]        # fallbacks are subtitle-only

    def make(idx):
        ln = script[idx]
        std = tts_dir / f"{idx:04d}.wav"
        stamp = tts_dir / f"{idx:04d}.txt"
        emo = ln.get("emo", "") if use_emotion else ""
        primary = voice_map[ln["speaker"]]
        # Attempt order. Keep the character's OWN voice as long as possible:
        # a line that fails WITH an emotion cue is retried on the SAME voice
        # WITHOUT the cue (some voices choke on the cue) before we ever switch to
        # a backup voice. So a failure costs the emotion on that line, not the
        # character's voice identity.
        fb = fallbacks.get(ln["speaker"], "")
        attempts = [(primary, emo)]
        if emo:
            attempts.append((primary, ""))          # same voice, drop the cue
        if fb and fb != primary:
            attempts.append((fb, emo))
            if emo:
                attempts.append((fb, ""))
        # cache is valid if made with any accepted voice, with or without a cue
        acc = {primary} | ({fb} if fb else set())
        valid = {ln["zh"] + "|" + v + "|" + e + "|" + cache_provider
                 for v in acc if v for e in {emo, ""}}
        if std.exists() and stamp.exists() and stamp.read_text() in valid:
            return idx, std, False
        raw = tts_dir / f"{idx:04d}.mp3"
        err = None
        for voice, e in attempts:
            if not voice:
                continue
            for attempt in range(2):
                try:
                    provider.synthesize(ln["zh"], voice, raw,
                                        api_key=fish_key, emotion=e)
                    to_std_wav(raw, std)
                    raw.unlink(missing_ok=True)
                    stamp.write_text(ln["zh"] + "|" + voice + "|" + e + "|" + cache_provider)
                    return idx, std, voice != primary   # True only if voice switched
                except Exception as ex:  # noqa: BLE001
                    err = ex
                    time.sleep(0.8 * (attempt + 1))
        print(f"  ! line {idx} voice failed: {str(err)[:150]}")
        return idx, None, False

    clips = {}
    from collections import Counter as _Counter
    fb_by_speaker = _Counter()
    workers = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(make, i) for i in todo]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            idx, path, used_fb = fut.result()
            if path:
                clips[idx] = path
                if used_fb:
                    fb_by_speaker[script[idx]["speaker"]] += 1
            done += 1
            if done >= 8 and not clips:
                for f in futs:
                    f.cancel()
                raise RuntimeError("Every line failed to generate audio -- "
                                   "see the Terminal window for the reason.")
            if progress:
                progress(done / max(len(todo), 1),
                         desc=f"Recording voices... {done} of {len(todo)}")
    if fb_by_speaker:
        who = ", ".join(f"{sp} ({c})" for sp, c in fb_by_speaker.most_common())
        DIAG.append(f"{sum(fb_by_speaker.values())} line(s) used a backup voice "
                    f"because the assigned voice kept failing: {who}. If one "
                    f"character dominates, their voice id is likely unreliable.")
    return clips


def apply_tempo(src, tempo, dst):
    if tempo <= 1.01:
        Path(dst).write_bytes(Path(src).read_bytes())
        return
    run_ffmpeg(["-i", str(src), "-filter:a", f"atempo={tempo:.4f}", str(dst)],
               "timing adjustment")


def assemble_ducked(orig_full_wav, placements, bg_hi, out_wav):
    """Original audio stays at bg_hi between lines and ducks to BG_LOW under
    Chinese speech, with short ramps. Chinese clips guaranteed non-overlapping."""
    orig, sr = sf.read(str(orig_full_wav), dtype="float32", always_2d=False)
    if orig.ndim > 1:
        orig = orig.mean(axis=1)
    n = len(orig)
    gain = np.full(n, float(bg_hi), dtype="float32")
    ramp = max(int(RAMP_S * sr), 1)
    clips_data = []
    for start, wav in placements:
        clip, _ = sf.read(str(wav), dtype="float32", always_2d=False)
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        pos = int(start * sr)
        end = min(pos + len(clip), n)
        if pos >= n:
            continue
        clips_data.append((pos, clip[:end - pos]))
        lo = max(pos - ramp, 0)
        hi = min(end + ramp, n)
        gain[lo:hi] = np.minimum(gain[lo:hi], BG_LOW)
    # smooth the gain curve to avoid clicks
    if n:
        k = np.ones(ramp, dtype="float32") / ramp
        gain = np.convolve(gain, k, mode="same").astype("float32")
    canvas = orig * gain
    for pos, clip in clips_data:
        canvas[pos:pos + len(clip)] += clip * 0.95
    np.clip(canvas, -1.0, 1.0, out=canvas)
    sf.write(str(out_wav), canvas, sr)


def find_cjk_font():
    candidates = [
        ("/System/Library/Fonts/PingFang.ttc", "PingFang SC"),
        ("/System/Library/Fonts/Hiragino Sans GB.ttc", "Hiragino Sans GB"),
        ("/System/Library/Fonts/STHeiti Medium.ttc", "Heiti SC"),
        ("/System/Library/Fonts/STHeiti Light.ttc", "Heiti SC"),
        ("/System/Library/Fonts/Supplemental/Songti.ttc", "Songti SC"),
        ("/Library/Fonts/Arial Unicode.ttf", "Arial Unicode MS"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "WenQuanYi Zen Hei"),
        ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei"),
        ("C:/Windows/Fonts/simhei.ttf", "SimHei"),
    ]
    for path, family in candidates:
        if Path(path).exists():
            return path, family
    return None, None


def mux(video, dub_wav, srt_path, out_mp4, burn=True):
    disclosure = ["-metadata", "comment=AI-translated and AI-dubbed with Peiyin",
                  "-metadata:s:a:0", "title=AI-generated Mandarin dub"]
    if burn:
        font_path, font_family = find_cjk_font()
        if not font_path:
            print("  ! no Chinese font found; embedding soft subtitles instead.")
        else:
            sp = str(srt_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            fd = str(Path(font_path).parent).replace(":", "\\:").replace("'", "\\'")
            style = f"FontName={font_family},FontSize=20,Outline=1,Shadow=0,MarginV=22"
            vf = f"subtitles='{sp}':fontsdir='{fd}':force_style='{style}'"
            try:
                run_ffmpeg(["-i", str(video), "-i", str(dub_wav),
                            "-map", "0:v:0", "-map", "1:a:0", "-vf", vf,
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "192k", "-shortest",
                            *disclosure, str(out_mp4)], "final export (burned subtitles)")
                print(f"  subtitles burned with font: {font_family}")
                return
            except RuntimeError as e:
                print("  ! burning failed, using soft subtitles:", str(e)[:200])
    try:
        run_ffmpeg(["-i", str(video), "-i", str(dub_wav), "-i", str(srt_path),
                    "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-c:s", "mov_text", "-metadata:s:s:0", "language=chi",
                    "-disposition:s:0", "default", "-shortest", *disclosure, str(out_mp4)],
                   "final export")
    except RuntimeError:
        run_ffmpeg(["-i", str(video), "-i", str(dub_wav),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", *disclosure, str(out_mp4)], "final export")


def _fit_key(src_wav, tempo):
    """Content-addressed name for a tempo-fitted clip.

    The fitted-clip cache used to be keyed by LINE INDEX + tempo
    ("0047_1.0000.wav"). Line indices are not stable: adding, removing or
    re-ordering a single line shifts every index after it, while most lines sit
    at tempo exactly 1.0000 -- so a re-run happily reused the PREVIOUS run's clip
    for a completely different line (wrong voice, wrong words, and a duration
    that no longer matched what the planner assumed, which also broke placement).
    Keying on the CONTENT of the source clip instead makes a wrong hit
    impossible
    stale files simply stop being referenced.
    """
    src = Path(src_wav)
    stamp = src.with_suffix(".txt")      # "zh|voice|emo|provider" from synth_raw
    try:
        ident = stamp.read_text()
    except Exception:  # noqa: BLE001
        try:
            h = hashlib.sha1()
            with open(src, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            ident = h.hexdigest()
        except Exception:  # noqa: BLE001
            ident = str(src.resolve())
    return hashlib.sha1(f"{ident}|{tempo:.4f}".encode("utf-8")).hexdigest()[:20]
