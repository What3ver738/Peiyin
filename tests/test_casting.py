"""Casting: per-profile storage, the opt-in public pool, and the guest rules."""

import json

import pytest

from peiyin import profiles, tts
from peiyin.config import CONFIG_PATH, DIAG, load_config
from peiyin.profiles import main_cast
from peiyin.tts import build_fish_voice_map, ensure_guest_voices


@pytest.fixture
def cast():
    c = {n: f"{i}" * 32 for i, n in enumerate(profiles.main_cast(), start=1)}
    tts.save_voice_cast(c)
    return c


def _set_pool(male=(), female=()):
    lines = [f"M {v}" for v in male] + [f"F {v}" for v in female]
    cfg = load_config()
    cfg["fish_guest_pool_text"] = "\n".join(lines)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")


def test_no_voice_ids_ship_by_default():
    """Hard rule 2: a fresh install casts nobody."""
    tts.save_voice_cast({})
    assert tts.load_voice_cast() == {}
    assert build_empty_map_has_no_ids()


def build_empty_map_has_no_ids():
    vm = tts.build_fish_voice_map()
    return all(not v for v in vm.values())


def test_cast_is_saved_per_profile(cast):
    """Two profiles must not share a cast: the same character name in a
    different show is a different person with a different voice."""
    assert tts.load_voice_cast() == cast
    path = profiles.create_profile("casting-other", show_name="Other Show")
    profiles.set_active_profile("casting-other")
    assert tts.load_voice_cast() == {}          # the other profile is unaffected
    tts.save_voice_cast({"Alice": "f" * 32})
    profiles.set_active_profile("example_show")
    assert tts.load_voice_cast() == cast        # ...and ours survived
    assert path.exists()


def test_public_pool_is_off_by_default():
    """Hard rule 3: the public voice library is opt-in, never assumed."""
    assert tts.auto_public_voices_enabled() is False
    assert tts.load_fish_auto_pool() == ([], [])
    # even with a key, nothing is fetched while the user has not opted in
    assert tts.ensure_fish_auto_pool("pretend-key") == ([], [])


def test_opting_in_is_remembered_per_profile():
    tts.set_auto_public_voices(True)
    assert tts.auto_public_voices_enabled() is True
    profiles.create_profile("casting-optin", show_name="Opt In")
    profiles.set_active_profile("casting-optin")
    assert tts.auto_public_voices_enabled() is False
    profiles.set_active_profile("example_show")
    assert tts.auto_public_voices_enabled() is True
    tts.set_auto_public_voices(False)


def test_a_guest_never_gets_a_cast_voice(cast):
    """The rule kept from the original: even if a cast voice is pasted into the
    guest pool, no guest may be given it."""
    _set_pool(male=[cast[main_cast()[0]], "a" * 32], female=[cast[main_cast()[1]], "b" * 32])
    tts.save_profile_setting(guest_voices={},
                             guest_genders={"Gil": "MALE", "Mia": "FEMALE"})
    got = tts.ensure_guest_voices(["Gil", "Mia"], "")
    assert got["Gil"] not in cast.values()
    assert got["Mia"] not in cast.values()
    assert got["Gil"] == "a" * 32 and got["Mia"] == "b" * 32


def test_guest_assignment_is_stable(cast):
    _set_pool(male=["a" * 32, "c" * 32], female=["b" * 32])
    tts.save_profile_setting(guest_voices={}, guest_genders={"Gil": "MALE"})
    first = tts.ensure_guest_voices(["Gil"], "")["Gil"]
    assert tts.ensure_guest_voices(["Gil"], "")["Gil"] == first


def test_empty_pool_leaves_the_guest_unvoiced_rather_than_borrowing(cast):
    """With nothing to draw from, a guest gets no voice at all -- the dub then
    stops with a clear message instead of impersonating a main character."""
    _set_pool()
    tts.save_profile_setting(guest_voices={}, guest_genders={"Gil": "MALE"})
    got = tts.ensure_guest_voices(["Gil"], "")
    assert got.get("Gil", "") == ""
    assert got.get("Gil") not in cast.values()
    # and the empty assignment is not cached, so it recovers once a pool exists
    assert "Gil" not in (tts.profile_setting("guest_voices") or {})


def test_main_cast_members_are_never_treated_as_guests(cast):
    assert tts.ensure_guest_voices(list(profiles.main_cast()), "") == {}


def test_preview_needs_a_key_and_a_voice():
    with pytest.raises(RuntimeError, match="Fish Audio API key"):
        tts.preview_voice("", "a" * 32)
    with pytest.raises(RuntimeError, match="No voice id"):
        tts.preview_voice("key", "")


def test_preview_line_falls_back_for_an_untested_language():
    assert "你好" in tts.preview_line("zh-CN")
    assert tts.preview_line("de-DE") == tts.DEFAULT_PREVIEW_TEXT


def test_search_without_a_key_returns_nothing():
    assert tts.search_fish_voices("") == []


# --------------------------------------------------------------------------
# Ported from the original app's --selftest: the end-to-end guest-voice rules.
# --------------------------------------------------------------------------

def test_cast_voices_are_reserved_and_guests_never_get_one():
    # --- Fish: cast voices are reserved; pasted guests are gender-matched and
    #     never a cast voice
    # Peiyin ships no voice ids: the cast is whatever the user chose. These are
    # obviously-fake ids invented for the test.
    CAST = {r: f"{i}" * 32 for i, r in enumerate(main_cast(), start=1)}

    def _setup(pool_text="", auto=None, genders=None):
        """Casting is per profile; the raw voice pool is global."""
        cfgk = load_config()
        cfgk["fish_guest_pool_text"] = pool_text
        if auto is None:
            cfgk.pop("fish_auto_guest_pool", None)
        else:
            cfgk["fish_auto_guest_pool"] = auto
        CONFIG_PATH.write_text(json.dumps(cfgk, ensure_ascii=False), encoding="utf-8")
        tts.set_auto_public_voices(auto is not None)
        tts.save_voice_cast(CAST)
        tts.save_profile_setting(guest_voices={}, guest_genders=genders or {})
    _setup("M 1111111111111111111111111111aaaa\n"
           "M 4444444444444444444444444444dddd\n"
           "F 2222222222222222222222222222bbbb\n",
           genders={"Wren": "MALE", "Iris": "FEMALE"})
    vm = build_fish_voice_map()
    for r in main_cast():
        assert vm[r] == CAST[r]
    assert vm["OtherMale"] == "1111111111111111111111111111aaaa"
    assert vm["OtherFemale"] == "2222222222222222222222222222bbbb"
    gv = ensure_guest_voices(["Wren", "Iris"], "")
    assert gv["Iris"] == "2222222222222222222222222222bbbb"
    assert gv["Wren"] not in CAST.values()
    assert ensure_guest_voices(["Wren"], "")["Wren"] == \
        gv["Wren"]                                              # stable

    # --- Fish: even if a LEAD's id is pasted into the guest pool, it must never
    # be used as a bucket or a guest voice.
    _setup("M " + CAST[main_cast()[0]] + "\n"                   # a cast voice, sneaked in
           "M 5555555555555555555555555555eeee\n"
           "F 6666666666666666666666666666ffff\n",
           genders={"Wren": "MALE"})
    vm_r = build_fish_voice_map()
    assert vm_r["OtherMale"] == "5555555555555555555555555555eeee"   # not Mara
    assert vm_r["OtherMale"] not in CAST.values()
    gvr = ensure_guest_voices(["Wren"], "")
    assert gvr["Wren"] not in CAST.values()

    # --- Fish: no paste, but a cached PUBLIC pool -> guests use it, never a lead
    _setup("", auto={
        "MALE": ["aaaa1111111111111111111111111111",
                 "bbbb2222222222222222222222222222"],
        "FEMALE": ["cccc3333333333333333333333333333"]},
        genders={"Wren": "MALE", "Iris": "FEMALE"})
    gv2 = ensure_guest_voices(["Wren", "Iris"], "")
    assert gv2["Wren"] in ("aaaa1111111111111111111111111111",
                              "bbbb2222222222222222222222222222")
    assert gv2["Iris"] == "cccc3333333333333333333333333333"
    assert gv2["Wren"] not in CAST.values()
    assert gv2["Iris"] not in CAST.values()

    # --- Fish: no paste AND no public pool -> guest gets NO voice ("") and is
    # NEVER a lead. Other* is also empty, never a lead. (The dub-time guard turns
    # this "" into a clear error rather than borrowing a main.)
    _setup("", genders={"Wren": "MALE"})
    vm_empty = build_fish_voice_map()
    assert vm_empty["OtherMale"] == "" and vm_empty["OtherFemale"] == ""
    assert vm_empty["OtherMale"] not in CAST.values()   # trivially, "" isn't
    gv3 = ensure_guest_voices(["Wren"], "")
    assert gv3.get("Wren", "") == ""          # no voice, and NOT cached
    assert gv3.get("Wren") not in CAST.values()
    DIAG.clear()
    _setup("")
    tts.save_voice_cast({})
