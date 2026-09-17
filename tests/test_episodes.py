"""Episode identification and transcript loading.

Fixtures are synthetic: three short invented "episodes" written for this test.
No real show's dialogue appears anywhere in this repository.
"""

import pytest

from peiyin import profiles
from peiyin.episodes import _cache_dir, ensure_scripts, episode_codes, identify_episode

# Three invented episodes. Each has its own distinctive vocabulary so a correct
# match is unambiguous, plus a little shared small-talk so the task is not
# trivially easy.
EPISODES = {
    "0101": [
        "Alice: The lighthouse keeper left the lamp burning all winter.",
        "Bob: Nobody has climbed those basalt stairs since the harbour froze.",
        "Alice: I counted eleven gulls nesting in the lantern room.",
        "Bob: Hello. How are you today.",
        "Alice: The foghorn needs a new diaphragm before the equinox.",
        "Bob: The supply boat is three days late again this month.",
        "Alice: I have been rationing the paraffin since Tuesday.",
        "Bob: Write it in the log before you forget the exact hour.",
    ],
    "0102": [
        "Alice: My sourdough starter has doubled again on the windowsill.",
        "Bob: You named it after your grandmother's terrier, which is strange.",
        "Alice: Rye flour makes it sour much faster than spelt does.",
        "Bob: Hello. How are you today.",
        "Alice: I am entering it in the county baking fair on Saturday.",
        "Bob: You should enter the rye loaf, not the plain white one.",
        "Alice: The judges last year preferred an open crumb structure.",
        "Bob: Then proof it overnight in the cold pantry instead.",
    ],
    "0103": [
        "Alice: The telescope mirror arrived with a chip on its edge.",
        "Bob: Grinding a new parabola takes months of patient polishing.",
        "Alice: Jupiter's moons were crisp through the eyepiece last night.",
        "Bob: Hello. How are you today.",
        "Alice: We should drive out past the reservoir to escape the streetlights.",
        "Bob: The reservoir road washes out whenever it rains hard.",
        "Alice: We can carry the mount in pieces and assemble it there.",
        "Bob: Bring the red torch so we do not ruin our night vision.",
    ],
}


@pytest.fixture
def transcript_folder(tmp_path):
    """A profile whose transcripts are three synthetic episodes on disk."""
    folder = tmp_path / "transcripts"
    folder.mkdir()
    for code, lines in EPISODES.items():
        (folder / f"{code}.txt").write_text("\n".join(lines), encoding="utf-8")

    try:
        path = profiles.create_profile("episode-test", show_name="Episode Test")
    except profiles.ProfileError:
        path = profiles.profile_paths()["episode-test"]
    text = (profiles.BUNDLED_PROFILE_DIR / "TEMPLATE.toml").read_text(encoding="utf-8")
    text = text.replace('local_folder = ""', f'local_folder = "{folder.as_posix()}"')
    path.write_text(text, encoding="utf-8")
    profiles.set_active_profile("episode-test")
    return folder


def _heard(code, noise=0):
    """What the transcriber produced: the episode's words, optionally degraded.

    `noise` garbles every Nth word, standing in for the words a transcriber
    mis-hears over music or a laugh track. Matching is trigram-based, so this
    is a real test: each garbled word destroys the three trigrams spanning it,
    and only the intact runs between them can carry the match.
    """
    segs = []
    for i, line in enumerate(EPISODES[code]):
        text = line.split(":", 1)[1].strip()
        if noise:
            words = text.split()
            text = " ".join("mmm" if k % noise == noise - 1 else w
                            for k, w in enumerate(words))
        segs.append({"start": i * 2.0, "end": i * 2.0 + 1.8, "en": text})
    return segs


def test_local_transcripts_are_read_and_cached(transcript_folder):
    have = ensure_scripts()
    assert sorted(have) == ["0101", "0102", "0103"]
    cached = list(_cache_dir(profiles.active()).glob("*.json"))
    assert len(cached) == 3


def test_the_right_episode_is_picked(transcript_folder):
    ensure_scripts()
    for code in EPISODES:
        ep, lines, score, runner = identify_episode(_heard(code))
        assert ep == code, (code, ep, score)
        assert score > runner


def test_the_right_episode_is_still_picked_from_noisy_input(transcript_folder):
    """Every fifth word mis-heard must not change the answer."""
    ensure_scripts()
    for code in EPISODES:
        ep, lines, score, runner = identify_episode(_heard(code, noise=5))
        assert ep == code, (code, ep, score)
        # and the winner is clearly ahead of the runner-up
        assert score >= runner * 1.6, (code, score, runner)


def test_unrelated_audio_scores_below_the_match_threshold(transcript_folder):
    """Something that is not any of these episodes must not be claimed as one.

    0.12 is the threshold `speakers.cast_speakers` uses to decide whether to
    trust the match; below it, casting falls back to voice recognition and
    records a warning instead of assigning the wrong cast.
    """
    ensure_scripts()
    unrelated = [
        {"start": 0.0, "end": 2.0,
         "en": "quarterly logistics throughput exceeded forecast in the "
               "southern distribution corridor"},
        {"start": 2.0, "end": 4.0,
         "en": "please submit the revised depreciation schedule before the "
               "audit committee convenes"},
    ]
    ep, lines, score, runner = identify_episode(unrelated)
    assert score < 0.12, (ep, score)


def test_an_ambiguous_match_is_not_clear_cut(transcript_folder):
    """Input that is half one episode and half another should not look
    confident: `cast_speakers` warns when the winner is under 1.6x the
    runner-up, and this is exactly that case."""
    ensure_scripts()
    mixed = _heard("0101")[:3] + _heard("0102")[:3]
    ep, lines, score, runner = identify_episode(mixed)
    assert runner > 0
    assert score / runner < 1.6, (score, runner)


def test_identify_needs_something_to_work_with(transcript_folder):
    ensure_scripts()
    with pytest.raises(RuntimeError, match="nothing was transcribed"):
        identify_episode([])


def test_a_profile_with_no_source_refuses_rather_than_guessing():
    """The example profile ships with no transcript source."""
    profiles.set_active_profile("example_show")
    with pytest.raises(RuntimeError, match="no transcript source"):
        ensure_scripts()


def test_episode_codes_come_from_the_profile():
    profiles.set_active_profile("example_show")
    codes = episode_codes()
    # 34 episodes across 3 seasons, minus the 2 ids folded into 1 multi-part
    # page, plus that page = 33 things to fetch
    assert len(codes) == 34 - 2 + 1
    assert "0101" in codes and "0211" not in codes and "0211-0212" in codes


def test_transcripts_are_cached_per_profile(transcript_folder):
    """Two shows must never see each other's transcripts."""
    ensure_scripts()
    mine = _cache_dir(profiles.active())
    profiles.set_active_profile("example_show")
    assert _cache_dir(profiles.active()) != mine
