"""Small dependency-free helpers shared across modules.

ffmpeg invocation, SRT timestamp formatting and writing, CJK line wrapping,
wav duration/conversion, cache keys, opening a folder in the OS file browser.
"""

import hashlib
import os
import subprocess
from pathlib import Path

import soundfile as sf

from .config import FFMPEG, SR


def run_ffmpeg(args, desc="ffmpeg step"):
    p = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error"] + args,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{desc} failed:\n{p.stderr[-1500:]}")


def reveal_folder(path):
    """Open the OS file browser at `path` so finished files are impossible to
    miss. Best-effort and silent on failure (e.g. a headless machine)."""
    try:
        import sys
        p = str(path)
        if sys.platform == "darwin":
            subprocess.Popen(["open", p])
        elif sys.platform.startswith("win"):
            os.startfile(p)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", p])
    except Exception:
        pass


def file_key(path):
    # Stream the complete file: equal names/sizes must not alias different media.
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def srt_time(t):
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


CJK_MAX = 18
CJK_BREAK = "，。！？、；：,.!?;… "


def wrap_cjk(text, width=CJK_MAX):
    text = text.strip()
    if len(text) <= width:
        return [text]
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= width:
            # prefer to break just after punctuation near the limit
            cut = len(cur)
            for k in range(len(cur) - 1, max(len(cur) - 6, 0), -1):
                if cur[k - 1] in CJK_BREAK:
                    cut = k
                    break
            lines.append(cur[:cut].strip())
            cur = cur[cut:]
    if cur.strip():
        lines.append(cur.strip())
    return [ln for ln in lines if ln]


def write_srt(entries, path):
    out, n = [], 0
    for e in entries:
        lines = wrap_cjk(e["zh"])
        # if it needs more than 2 screen lines, split into timed sub-entries
        chunks = [lines[i:i + 2] for i in range(0, len(lines), 2)] or [[""]]
        span = max(e["end"] - e["start"], 0.4)
        per = span / len(chunks)
        for c, chunk in enumerate(chunks):
            st = e["start"] + c * per
            en = st + per
            n += 1
            out += [str(n), f"{srt_time(st)} --> {srt_time(en)}",
                    "\n".join(chunk), ""]
    Path(path).write_text("\n".join(out), encoding="utf-8")


def wav_duration(path):
    i = sf.info(str(path))
    return i.frames / i.samplerate


def to_std_wav(src, dst):
    run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), str(dst)],
               "audio conversion")
