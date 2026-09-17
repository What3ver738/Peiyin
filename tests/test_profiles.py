"""Profile loading, validation, and the transcript-format parsers."""

import pytest

from peiyin import profiles
from peiyin.episodes import (
    PARSERS,
    canonical_speaker,
    parse_name_colon,
    parse_srt_with_speakers,
)
from peiyin.profiles import ProfileError, load_profile, parse_profile

MINIMAL = {
    "name": "Test show",
    "characters": [
        {"name": "Alice", "aliases": ["Ali", "Alice Smith"], "gender": "F",
         "main": True},
        {"name": "Bob", "aliases": ["Bobby", "Robert Jones"], "gender": "M",
         "main": True},
        {"name": "Carol", "gender": "F", "main": False},
    ],
}


def test_valid_profile_derives_the_cast():
    p = parse_profile(MINIMAL)
    assert p.main_cast == ["Alice", "Bob"]
    assert p.speakers == ["Alice", "Bob", "OtherMale", "OtherFemale"]
    assert p.labels[-1] == "Skip"
    assert p.male_roles == {"Bob", "OtherMale"}


def test_aliases_split_by_word_count():
    """One word is a nickname, several are a full name -- they are matched
    differently by canonical_speaker, so they must land in different maps."""
    p = parse_profile(MINIMAL)
    assert p.aliases == {"alice": "Alice", "ali": "Alice",
                         "bob": "Bob", "bobby": "Bob"}
    assert p.full_names == {"alice smith": "Alice", "robert jones": "Bob"}


def test_non_main_characters_are_not_in_the_cast():
    p = parse_profile(MINIMAL)
    assert "Carol" not in p.main_cast
    assert "carol" not in p.aliases


@pytest.mark.parametrize("bad, message", [
    ({"characters": [{"name": "A", "main": True}]}, "name"),
    ({"name": "X", "characters": []}, "main characters"),
    ({"name": "X", "characters": [{"main": True}]}, "needs a `name`"),
    ({"name": "X", "characters": [{"name": "A", "main": True},
                                  {"name": "a", "main": True}]}, "duplicate"),
    ({"name": "X", "characters": [{"name": "Skip", "main": True}]}, "reserved"),
    ({"name": "X", "characters": [{"name": "A", "main": True, "gender": "yes"}]},
     "gender"),
    ({"name": "X", "characters": [{"name": "A", "main": True}],
      "transcripts": {"format": "nope"}}, "unknown transcript format"),
    ({"name": "X", "characters": [{"name": "A", "main": True}],
      "theme": {"skip_intro_seconds": -5}}, "negative"),
])
def test_invalid_profiles_are_rejected(bad, message):
    with pytest.raises(ProfileError) as e:
        parse_profile(bad)
    assert message in str(e.value)


def test_no_transcript_source_means_no_transcript_mode(tmp):
    p = parse_profile(MINIMAL)
    assert p.has_transcripts is False
    p2 = parse_profile({**MINIMAL, "transcripts": {"local_folder": "/tmp/x"}})
    assert p2.has_transcripts is True


def test_example_profile_ships_no_transcript_source_and_no_lyrics():
    """Hard rules: no bundled transcript source, nothing lyric-derived."""
    p = load_profile(profiles.BUNDLED_PROFILE_DIR / "example_show.toml")
    assert p.transcripts.local_folder == ""
    assert p.transcripts.url_list == ()
    assert p.transcripts.url_base == ()
    assert p.has_transcripts is False
    assert p.theme.theme_words == ()


def test_template_is_loadable_so_create_profile_produces_a_valid_file():
    p = load_profile(profiles.BUNDLED_PROFILE_DIR / "TEMPLATE.toml")
    assert p.main_cast == ["Alice", "Bob"]


def test_create_profile_writes_an_editable_copy():
    path = profiles.create_profile("my-test-show", show_name="My Test Show")
    assert path.exists()
    assert load_profile(path).name == "My Test Show"
    with pytest.raises(ProfileError):
        profiles.create_profile("my-test-show")       # no silent overwrite


# --------------------------------------------------------------- parsers

def test_every_known_format_has_a_parser():
    assert set(PARSERS) == set(profiles.KNOWN_FORMATS)


def test_name_colon_parser():
    lines = parse_name_colon(
        "Alice: The garden gate is blue.\n"
        "Bob: The gardener painted it last week.\n"
        "and the paint is finally dry.\n"      # wrapped continuation
        "\n"
        "Carol: Hi.\n")
    assert [x["speaker"] for x in lines] == ["Alice", "Bob", "Carol"]
    assert lines[1]["text"].endswith("finally dry.")


def test_srt_with_speakers_parser():
    lines = parse_srt_with_speakers(
        "1\n00:00:01,000 --> 00:00:03,000\nAlice: The garden gate\n\n"
        "2\n00:00:03,100 --> 00:00:05,000\nis blue.\n\n"
        "3\n00:00:05,500 --> 00:00:07,000\nBob: Welcome.\n")
    assert [x["speaker"] for x in lines] == ["Alice", "Bob"]
    # a continued cue joins the utterance it belongs to
    assert lines[0]["text"] == "The garden gate is blue."


def test_canonical_speaker_uses_the_active_profile():
    """The example profile is active (see conftest), so its aliases resolve and
    relative/look-alike markers still fall through to GUEST."""
    assert canonical_speaker("Mara Aldridge:") == "Mara"
    assert canonical_speaker("Pheebs") == "GUEST"          # not in this profile
    assert canonical_speaker("Nads") == "Nadia"
    assert canonical_speaker("Oskar's Grandmother") == "GUEST"
    assert canonical_speaker("Fake Priya") == "GUEST"
    assert canonical_speaker("All") == "GROUP"


def test_switching_profile_changes_the_cast():
    path = profiles.create_profile("switch-test", show_name="Switch Test")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('name = "Alice"', 'name = "Zara"'), encoding="utf-8")
    profiles.set_active_profile("switch-test")
    assert profiles.main_cast() == ["Zara", "Bob"]
    assert canonical_speaker("Zara") == "Zara"
    assert canonical_speaker("Mara") == "GUEST"     # not in this profile
