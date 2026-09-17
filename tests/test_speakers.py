"""Speaker attribution: alignment, the script-first spine, clustering."""

from collections import Counter

import numpy as np

from peiyin.episodes import canonical_speaker
from peiyin.pipeline import merge_by_script_line
from peiyin.speakers import (
    apply_cluster_majority,
    cluster_voices,
    insert_missing_script_lines,
    place_fallback_subtitles,
    script_line_spans,
)
from peiyin.transcribe import merge_utterances


def test_voice_clustering_finds_one_cluster_per_speaker():
    # --- voice clustering + majority
    rng = np.random.default_rng(1)
    def unit(v): return v / np.linalg.norm(v)
    base = [unit(rng.standard_normal(64).astype("float32")) for _ in range(3)]
    embs, truthv = {}, {}
    for j in range(60):
        b = j % 3
        wins = []
        for _ in range(2):            # two windows per line, like real audio
            nz = rng.standard_normal(64).astype("float32")
            nz = 0.5 * nz / np.linalg.norm(nz)
            wins.append(unit(base[b] + nz))
        embs[j] = wins
        truthv[j] = b
    l2c, k = cluster_voices(embs)
    from collections import Counter
    for cid in range(k):
        idxs = [j for j, c in l2c.items() if c == cid]
        top = Counter(truthv[j] for j in idxs).most_common(1)[0]
        assert top[1] / len(idxs) >= 0.95
    lab = [["Alice", "Bob", "Dev"][truthv[j]] for j in range(60)]
    lab[0] = "Finn"
    lab = apply_cluster_majority(lab, l2c)
    assert lab[0] == "Alice"
    print(f"voice clustering OK ({k} clusters for 3 voices, majority fixes errors)")


def test_aliases_resolve_but_relatives_and_lookalikes_do_not(synthetic_profile):
    """A name or alias from the profile resolves to that character. A
    possessive or a relative/look-alike marker is a DIFFERENT person and must
    fall through to GUEST, or a grandmother gets her grandson's voice."""
    # exact names and nicknames
    assert canonical_speaker("Alice") == "Alice"
    assert canonical_speaker("Ali") == "Alice"
    assert canonical_speaker("ALICE SMITH") == "Alice"
    assert canonical_speaker("Alice (angrily)") == "Alice"
    assert canonical_speaker("Alice:") == "Alice"
    # relatives, look-alikes and possessives are other people
    assert canonical_speaker("Alice's Grandmother") == "GUEST"
    assert canonical_speaker("Fake Alice") == "GUEST"
    assert canonical_speaker("Young Alice") == "GUEST"
    assert canonical_speaker("Mrs Alice") == "GUEST"
    assert canonical_speaker("Alice's Date") == "GUEST"
    # groups and strangers
    assert canonical_speaker("Alice and Bob") == "GROUP"
    assert canonical_speaker("All") == "GROUP"
    assert canonical_speaker("Gil") == "GUEST"


def test_script_line_spans_locate_a_line_from_word_times():
    # --- script-first: word-level per-script-line spans locate a line from the
    #     Whisper WORD timestamps.
    wl = [{"en": "hello there everyone",
           "words": [[1.0, 1.3], [1.4, 1.7], [1.8, 2.2]]},
          {"en": "i am very happy",
           "words": [[3.0, 3.2], [3.3, 3.6], [3.7, 4.0], [4.1, 4.4]]}]
    sl2 = [{"speaker": "Mara", "text": "hello there everyone"},
           {"speaker": "Oskar", "text": "i am very happy today"}]
    sps = script_line_spans(wl, sl2)
    assert 0 in sps and abs(sps[0][0] - 1.0) < 0.2, sps
    assert 1 in sps and abs(sps[1][0] - 3.0) < 0.3, sps


def test_every_script_line_is_present_exactly_once():
    # --- script-first: every script line ends up present exactly once; a missing
    #     line becomes a subtitle-only fallback placed between anchored neighbours.
    sfl = [{"speaker": "Mara", "text": "one two three four"},
           {"speaker": "Oskar", "text": "five six seven eight"},   # MISSING
           {"speaker": "Mara", "text": "nine ten eleven twelve"}]
    dub = [{"start": 0.0, "end": 1.0, "sidx": 0, "speaker": "Mara", "zh": "a",
            "emo": "", "words": [[0.0, 1.0]]},
           {"start": 5.0, "end": 6.0, "sidx": 2, "speaker": "Mara", "zh": "c",
            "emo": "", "words": [[5.0, 6.0]]}]
    spans_t = {0: (0.0, 1.0, 4), 2: (5.0, 6.0, 4)}
    merged_s, added = insert_missing_script_lines([dict(x) for x in dub],
                                                  sfl, spans_t)
    assert added == 1, added
    fb = [ln for ln in merged_s if ln.get("fallback")]
    assert len(fb) == 1 and fb[0]["sidx"] == 1, fb
    assert 1.0 <= fb[0]["start"] <= 5.0, fb                 # in the empty span
    sc = Counter(ln.get("sidx") for ln in merged_s)         # no doubling
    assert sc[0] == 1 and sc[1] == 1 and sc[2] == 1, sc
    # low coverage -> spine not trusted -> no flooding with guesswork
    few = [{"speaker": "Mara", "text": f"word{k} extra token here"}
           for k in range(20)]
    dub_few = [{"start": 0.0, "end": 1.0, "sidx": 0, "speaker": "Mara",
                "zh": "a", "emo": "", "words": [[0.0, 1.0]]}]
    _, added2 = insert_missing_script_lines([dict(x) for x in dub_few], few,
                                            {0: (0.0, 1.0, 4)})
    assert added2 == 0, added2


def test_fallback_subtitles_land_between_placed_neighbours():
    # --- fallback subtitles land between the PLACED anchored neighbours and never
    #     into their audio; the anchored audio is untouched.
    scr_fb = [{"start": 0.0, "end": 1.0, "sidx": 0, "speaker": "Mara", "zh": "a"},
              {"start": 2.0, "end": 3.0, "sidx": 1, "speaker": "Oskar",
               "zh": "\u7f3a\u5931\u53f0\u8bcd", "fallback": True},
              {"start": 5.0, "end": 6.0, "sidx": 2, "speaker": "Mara", "zh": "c"}]
    placed_sidx = {0: (0.0, 1.0), 2: (5.0, 6.0)}
    fb_ent = place_fallback_subtitles(scr_fb, placed_sidx, [(1.2, 4.8)], 10.0)
    assert len(fb_ent) == 1, fb_ent
    assert 1.0 <= fb_ent[0]["start"] and fb_ent[0]["end"] <= 5.0, fb_ent


def test_fragments_of_one_line_are_merged():
    """The transcriber splits one utterance across several segments. Those
    fragments must come back as a single script line, keeping the speaker and
    the full text, or the dub says the same sentence twice."""
    segments = [
        {"start": 0.0, "end": 0.8, "en": "I have been thinking", "sidx": 0},
        {"start": 0.8, "end": 1.6, "en": "about the lighthouse", "sidx": 0},
        {"start": 1.6, "end": 2.4, "en": "all winter long", "sidx": 0},
        {"start": 3.0, "end": 3.9, "en": "So has everyone else", "sidx": 1},
    ]
    segs, labels, n_merged = merge_by_script_line(
        segments, ["Alice", "Alice", "Alice", "Bob"])
    assert len(segs) == 2, segs
    assert labels == ["Alice", "Bob"]
    assert segs[0]["en"] == ("I have been thinking about the lighthouse "
                             "all winter long")
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 2.4
    assert n_merged == 2          # three fragments collapsed into one line


def test_consecutive_lines_from_one_speaker_stay_separate_without_a_script():
    """With no script index to group by, two separate utterances stay two
    lines -- merging them would stretch one dub across a real pause."""
    segments = [
        {"start": 0.0, "end": 0.8, "en": "The lamp is lit"},
        {"start": 6.0, "end": 6.9, "en": "The lamp is out"},
    ]
    segs, labels, _ = merge_by_script_line(segments, ["Alice", "Alice"])
    assert len(segs) == 2
    assert labels == ["Alice", "Alice"]


def test_utterance_merging_respects_a_real_pause():
    """merge_utterances joins close fragments but never bridges a long gap."""
    close = merge_utterances([
        {"start": 0.0, "end": 1.0, "en": "the foghorn needs"},
        {"start": 1.1, "end": 2.0, "en": "a new diaphragm"},
    ])
    assert len(close) == 1
    far = merge_utterances([
        {"start": 0.0, "end": 1.0, "en": "the foghorn needs"},
        {"start": 9.0, "end": 10.0, "en": "a new diaphragm"},
    ])
    assert len(far) == 2
