"""Audio assembly, subtitles and muxing."""

import subprocess

import numpy as np
import soundfile as sf

from peiyin.config import FFMPEG, SR
from peiyin.mix import _fit_key, assemble_ducked, mux
from peiyin.util import write_srt


def test_assembly_ducks_the_original_under_the_dub(tmp):
    # --- ducked assembly
    orig = tmp / "orig.wav"
    t = np.linspace(0, 10, SR * 10, endpoint=False)
    sf.write(str(orig), (0.5 * np.sin(2 * np.pi * 200 * t)).astype("float32"), SR)
    clip = tmp / "clip.wav"
    tc = np.linspace(0, 1, SR, endpoint=False)
    sf.write(str(clip), (0.4 * np.sin(2 * np.pi * 600 * tc)).astype("float32"), SR)
    out = tmp / "mix.wav"
    assemble_ducked(orig, [(3.0, clip)], 0.30, out)
    mix, _ = sf.read(str(out), dtype="float32")
    seg_mid = mix[int(3.5 * SR):int(3.6 * SR)]
    seg_far = mix[int(7.0 * SR):int(7.1 * SR)]
    # under speech the 200Hz bed must be ~BG_LOW; away from speech ~0.30
    from numpy.fft import rfft, rfftfreq
    def level_at(seg, freq):
        sp = np.abs(rfft(seg))
        fr = rfftfreq(len(seg), 1 / SR)
        return sp[np.argmin(np.abs(fr - freq))]
    duck_ratio = level_at(seg_mid, 200) / level_at(seg_far, 200)
    assert duck_ratio < 0.5, duck_ratio
    print(f"assemble_ducked OK (bed ducks to {duck_ratio:.2f}x under speech)")


def test_srt_is_written_from_the_placed_times(tmp):
    # --- srt from placed times
    srt = tmp / "s.srt"
    write_srt([{"start": 1.0, "end": 2.5, "zh": "你好"}], srt)
    assert "00:00:01,000 --> 00:00:02,500" in srt.read_text(encoding="utf-8")


def test_mux_burns_subtitles_into_a_video(tmp):
    # --- mux with burned subs on a generated video
    v = tmp / "v.mp4"
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=320x240:d=6",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(v)], check=True)
    dub = tmp / "d.wav"
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=6",
                    "-ac", "1", "-ar", str(SR), str(dub)], check=True)
    srt = tmp / "burn.srt"
    write_srt([{"start": 1.0, "end": 2.5, "zh": "你好"}], srt)
    outv = tmp / "o.mp4"
    mux(v, dub, srt, outv, burn=True)
    assert outv.exists() and outv.stat().st_size > 1000


def test_the_tempo_fit_cache_is_content_addressed(tmp, tone_wav):
    # --- REGRESSION: the tempo-fit cache must never serve another line's audio
    #     when line indices shift (adding one line shifts every index after it,
    #     and most lines sit at tempo exactly 1.0000).
    fitd = tmp / "fitkey"
    fitd.mkdir(exist_ok=True)
    a_wav, b_wav = fitd / "0047.wav", fitd / "0099.wav"
    tone_wav(a_wav, 120, dur=1.9)
    tone_wav(b_wav, 250, dur=0.6)
    (fitd / "0047.txt").write_text("我们分手了|voiceA||fish", encoding="utf-8")
    (fitd / "0099.txt").write_text("闻闻小猫|voiceB||fish", encoding="utf-8")
    assert _fit_key(a_wav, 1.0) != _fit_key(b_wav, 1.0), "fit cache collision!"
    assert _fit_key(a_wav, 1.0) == _fit_key(a_wav, 1.0), "fit key unstable"
    assert _fit_key(a_wav, 1.0) != _fit_key(a_wav, 1.35), "tempo ignored"
    # different content at the SAME index must not collide either
    (fitd / "0047.txt").write_text("完全不同的台词|voiceC||fish", encoding="utf-8")
    assert _fit_key(a_wav, 1.0) != _fit_key(b_wav, 1.0)
