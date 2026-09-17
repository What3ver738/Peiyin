"""Voice selection, guest pools and emotion cues."""

import json

from peiyin.config import CONFIG_PATH, DIAG, load_config
from peiyin.transcribe import estimate_median_f0
from peiyin.tts import (
    MAX_GUEST_F0,
    _guess_voice_gender,
    _has_cjk,
    _is_bad_voice,
    clean_emotion,
    filter_pool_by_pitch,
    parse_guest_pool,
)


def test_emotion_cues_are_normalised():
    # --- Fish: emotion cue normalisation
    assert clean_emotion("[excited]") == "[excited]"
    assert clean_emotion("angry") == "[angry]"
    assert clean_emotion("(sad)") == "[sad]"
    assert clean_emotion("") == ""
    assert clean_emotion("x" * 50) == ""          # too long -> dropped


def test_guest_pool_parsing_reads_gender_and_id():
    # --- Fish: guest pool parsing (gender + hex id, ignores labels)
    pool = parse_guest_pool(
        "M 1111111111111111111111111111aaaa\n"
        "female 女大学生 2222222222222222222222222222bbbb\n"
        "女 3333333333333333333333333333cccc\n"
        "# a comment line\n"
        "garbage with no id\n")
    assert pool == [("MALE", "1111111111111111111111111111aaaa"),
                    ("FEMALE", "2222222222222222222222222222bbbb"),
                    ("FEMALE", "3333333333333333333333333333cccc")], pool


def test_voice_gender_is_guessed_from_the_title():
    # --- Fish: gender heuristics from a voice title (female checked first)
    assert _guess_voice_gender("温柔女声") == "FEMALE"
    assert _guess_voice_gender("磁性大叔") == "MALE"
    assert _guess_voice_gender("female voice") == "FEMALE"   # not misread MALE
    assert _guess_voice_gender("播音腔") == ""                # no gender word
    assert _has_cjk("女大学生") and not _has_cjk("english only")


def test_sexy_and_asmr_voices_are_denied():
    # --- sexy/ASMR voice denylist
    assert _is_bad_voice("温柔女声 ASMR 助眠") is True
    assert _is_bad_voice("性感御姐") is True
    assert _is_bad_voice("Sexy Girlfriend Whisper") is True
    assert _is_bad_voice("新闻联播 男声 主播") is False
    assert _is_bad_voice("纪录片 解说 男声") is False


def test_the_pitch_filter_rejects_a_shrill_guest_voice(tmp, tone_wav):
    # --- guest pitch filter: a shrill probe is rejected, a normal one kept. The
    #     same estimator drives measure_line_genders, so gender stays correct.
    hi_wav = tmp / "hi.wav"
    lo_wav = tmp / "lo.wav"
    tone_wav(hi_wav, 360)
    tone_wav(lo_wav, 180)
    fhi = estimate_median_f0(hi_wav)
    flo = estimate_median_f0(lo_wav)
    assert fhi and fhi > MAX_GUEST_F0, fhi
    assert flo and flo < MAX_GUEST_F0, flo
    # filter_pool_by_pitch drops a cached-shrill voice, keeps unknowns (fail open)
    cfgp = load_config()
    saved_f0 = cfgp.get("fish_voice_f0")
    cfgp["fish_voice_f0"] = {"shrill000": 340.0, "normal000": 200.0}
    CONFIG_PATH.write_text(json.dumps(cfgp, ensure_ascii=False))
    kept = filter_pool_by_pitch("", ["shrill000", "normal000", "unknown000"])
    assert kept == ["normal000", "unknown000"], kept
    cfgp = load_config()
    if saved_f0 is None:
        cfgp.pop("fish_voice_f0", None)
    else:
        cfgp["fish_voice_f0"] = saved_f0
    CONFIG_PATH.write_text(json.dumps(cfgp, ensure_ascii=False))
    DIAG.clear()
    print(f"pitch filter OK (shrill {fhi:.0f}Hz rejected, {flo:.0f}Hz kept)")
