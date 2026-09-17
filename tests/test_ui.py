"""The script review table.

The table is where a user corrects the machine's guesses before dubbing, so a
round trip must preserve edits and normalise a speaker name that was typed in
the wrong case -- against the ACTIVE profile's labels, not a fixed list.
"""

from peiyin.ui import rows_to_script, script_to_rows


def test_script_table_survives_a_round_trip(synthetic_profile):
    scr = [{"start": 0.0, "end": 1.0, "en": "hi", "speaker": "Alice", "zh": "嗨"},
           {"start": 1.0, "end": 2.0, "en": "yo", "speaker": "Alice", "zh": "哟"}]
    rows = script_to_rows(scr)
    assert rows[0][2] == "Alice" and rows[0][4] == "嗨"

    rows[1][2] = "bob"                 # user reassigns, typing the wrong case
    rows[1][4] = "改过的台词"            # ...and rewrites the line
    scr2 = rows_to_script(rows, [dict(x) for x in scr])
    assert scr2[1]["speaker"] == "Bob"
    assert scr2[1]["zh"] == "改过的台词"
    assert scr2[0]["speaker"] == "Alice"      # the untouched row is untouched


def test_an_unknown_speaker_is_ignored_rather_than_accepted(synthetic_profile):
    """A typo must not invent a character the profile has never heard of."""
    scr = [{"start": 0.0, "end": 1.0, "en": "hi", "speaker": "Alice", "zh": "嗨"}]
    rows = script_to_rows(scr)
    rows[0][2] = "Nobody"
    scr2 = rows_to_script(rows, [dict(x) for x in scr])
    assert scr2[0]["speaker"] == "Alice"


def test_clearing_the_translation_of_a_skipped_line_blanks_it(synthetic_profile):
    scr = [{"start": 0.0, "end": 1.0, "en": "la la la", "speaker": "Alice",
            "zh": "啦啦啦"}]
    rows = script_to_rows(scr)
    rows[0][2] = "Skip"
    rows[0][4] = ""
    scr2 = rows_to_script(rows, [dict(x) for x in scr])
    assert scr2[0]["speaker"] == "Skip" and scr2[0]["zh"] == ""
