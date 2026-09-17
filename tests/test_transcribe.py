"""Transcription helpers and speech/silence detection."""

import numpy as np
import soundfile as sf

from peiyin.transcribe import measure_line_genders, speech_silence_map, split_merged


def test_split_merged_breaks_a_merged_line_into_utterances():
    # --- merged-line splitting
    segs = split_merged({"start": 0.0, "end": 3.0,
                         "en": "- Hey Felix. - What's up? - Nothing."})
    assert len(segs) == 3, segs
    assert abs(segs[-1]["end"] - 3.0) < 0.01
    assert segs[0]["en"] == "Hey Felix."
    single = split_merged({"start": 0, "end": 1, "en": "Just one line."})
    assert len(single) == 1


def test_pitch_classifies_voice_gender(tmp):
    # --- pitch gender classifier
    srx = 16000
    def voiced(f0, dur):
        tt = np.linspace(0, dur, int(srx*dur), endpoint=False)
        x = sum((0.6/h)*np.sin(2*np.pi*f0*h*tt) for h in range(1, 6))
        return (0.3*x/np.max(np.abs(x))).astype("float32")
    wavp = tmp / "pitch.wav"
    sig = np.concatenate([voiced(110, 1.5), np.zeros(srx//2, dtype="float32"),
                          voiced(225, 1.5)])
    sf.write(str(wavp), sig, srx)
    g = measure_line_genders(wavp, [{"start": 0.0, "end": 1.5},
                                    {"start": 2.0, "end": 3.5}])
    assert g == ["MALE", "FEMALE"], g


def test_the_silence_map_finds_the_quiet_gap_not_the_speech(tmp):
    # --- v2: energy silence-map must find the quiet gap, not the speech
    srz = 16000
    def _tone(dur, f):
        tt = np.linspace(0, dur, int(srz * dur), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * f * tt)).astype("float32")
    sig = np.concatenate([_tone(2.0, 220),
                          np.zeros(int(srz * 1.0), dtype="float32"),
                          _tone(2.0, 300)])
    wsz = tmp / "sil.wav"
    sf.write(str(wsz), sig, srz)
    sm = speech_silence_map(wsz)
    assert len(sm) == 1, sm
    gs, ge = sm[0]
    assert 1.8 <= gs <= 2.25 and 2.75 <= ge <= 3.2, sm
    # words carrying real timestamps survive splitting AND merging
    ws_seg = split_merged({"start": 0.0, "end": 2.0,
                           "en": "- Hey. - What's up?",
                           "words": [[0.0, 0.4], [1.0, 1.6]]})
    assert len(ws_seg) == 2 and ws_seg[0]["words"] and ws_seg[1]["words"], ws_seg
    print(f"speech_silence_map OK (gap {gs:.2f}-{ge:.2f}s)")
