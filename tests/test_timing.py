"""The timing planner.

These are pure-function tests: no audio, no network, no config. Everything
the planner needs arrives as plain data, which is the point of keeping
`peiyin.timing` free of I/O.
"""

from peiyin.pipeline import merge_by_script_line
from peiyin.timing import (
    MAX_LAG,
    MAX_SPEEDUP,
    MIN_GAP,
    build_line_windows,
    extend_end_to_speech,
    plan_timeline,
    silence_from_regions,
)


def test_dense_overlapping_lines_come_out_non_overlapping():
    # --- planner: dense overlapping lines must come out non-overlapping
    script = [{"start": i * 1.0, "end": i * 1.0 + 0.9, "zh": "x", "speaker": "Alice"}
              for i in range(10)]
    durations = {i: 1.6 for i in range(10)}   # every clip longer than its slot
    plan = plan_timeline(script, durations, 60.0)
    # sort by placement, then the real guarantee: no two lines overlap
    prev_end = None
    for i, place, tempo in sorted(plan, key=lambda x: x[1]):
        assert tempo <= MAX_SPEEDUP + 1e-6
        if prev_end is not None:
            assert place >= prev_end - 1e-6, (place, prev_end)
        prev_end = place + durations[i] / tempo
    # drift must self-heal after a gap
    script2 = ([{"start": 0.0, "end": 0.5, "zh": "x", "speaker": "Alice"},
                {"start": 0.6, "end": 1.1, "zh": "x", "speaker": "Alice"},
                {"start": 30.0, "end": 31.0, "zh": "x", "speaker": "Alice"}])
    plan2 = plan_timeline(script2, {0: 2.0, 1: 2.0, 2: 1.0}, 60.0)
    assert abs(plan2[2][1] - 30.0) < 0.15, plan2  # 3rd line back on schedule


def test_a_line_plays_into_empty_time_at_normal_speed():
    # --- placement: each line starts ON its onset (never early) and plays its
    #     full length into the empty time after it, without needless speed-up.
    long_scr = [{"start": 0.0, "end": 20.0, "zh": "x", "speaker": "Alice"}]
    lp = plan_timeline(long_scr, {0: 5.0}, 60.0)
    assert lp[0][1] < 0.5 and lp[0][2] == 1.0, lp     # at the onset, normal speed
    short_scr = [{"start": 10.0, "end": 12.0, "zh": "x", "speaker": "Alice"}]
    sp = plan_timeline(short_scr, {0: 1.0}, 60.0)
    assert abs(sp[0][1] - 10.0) < 1e-6, sp            # exactly on its onset, never early
    late = [{"start": 50.0, "end": 70.0, "zh": "x", "speaker": "Alice"}]
    lpp = plan_timeline(late, {0: 4.0}, 120.0)
    assert abs(lpp[0][1] - 50.0) < 1e-6, lpp
    # KEY: a long clip with nobody talking after it is NOT sped up or cut short --
    # it starts on its onset and simply plays on into the empty time.
    gapscr = [{"start": 0.0, "end": 1.0, "zh": "x", "speaker": "Alice"},
              {"start": 20.0, "end": 21.0, "zh": "x", "speaker": "Bob"}]
    gp = plan_timeline(gapscr, {0: 6.0, 1: 1.0}, 40.0)
    assert abs(gp[0][1] - 0.0) < 1e-6, gp             # starts on its onset
    assert gp[0][2] == 1.0, gp                        # NOT sped up (was ~1.6 before)
    assert abs(gp[0][1] + 6.0 - 6.0) < 1e-6           # plays its full 6s (0..6) into the gap
    assert abs(gp[1][1] - 20.0) < 1e-6, gp            # next line still on its onset
    # but it must never run INTO the next line's audio: an absurdly long first
    # clip is sped up only as the last resort, and the next line never overlaps.
    gp3 = plan_timeline(gapscr, {0: 40.0, 1: 1.0}, 40.0)
    e0 = gp3[0][1] + 40.0 / gp3[0][2]
    assert gp3[1][1] >= e0 + MIN_GAP - 1e-6, gp3      # no overlap with the next line


def test_lag_is_bounded_never_early_and_resets_across_gaps():
    # --- dense dialogue: a line may lag up to MAX_LAG (drift preferred over
    #     speed-up), never overlaps, and the line after a real gap resets to its
    #     onset (the backlog drains, nothing crosses into the next scene).
    dense = [{"start": i * 1.5, "end": i * 1.5 + 1.0, "zh": "x", "speaker": "A"}
             for i in range(8)]
    dense.append({"start": 40.0, "end": 41.0, "zh": "x", "speaker": "A"})
    dd = {i: 2.2 for i in range(8)}
    dd[8] = 1.0
    pl = plan_timeline(dense, dd, 60.0)
    assert abs(pl[8][1] - 40.0) < 1e-6, pl               # resets on its onset after the gap
    for idx, place, tempo in pl:
        assert tempo <= MAX_SPEEDUP + 1e-6, (idx, tempo)
        assert place >= dense[idx]["start"] - 1e-6, (idx, place)          # never early
        assert place <= dense[idx]["start"] + MAX_LAG + 1e-6, (idx, place)  # lag bounded
    prev = None
    for idx, place, tempo in sorted(pl, key=lambda x: x[1]):
        if prev is not None:
            assert place >= prev - 1e-6                    # no overlap
        prev = place + dd[idx] / tempo


def test_a_long_speech_splits_at_a_real_pause():
    # --- long speech with a real pause splits into pieces; continuous stays one
    full = "Hello there everyone I am so happy to see you"
    seg_gap = [{"start": 0.0, "end": 2.0, "en": "Hello there everyone",
                "sidx": 5, "stext": full},
               {"start": 9.0, "end": 11.0, "en": "I am so happy to see you",
                "sidx": 5, "stext": full}]          # 7s pause between them
    o1, _, _ = merge_by_script_line(seg_gap, ["Alice", "Alice"])
    assert len(o1) == 2, o1                    # split on the pause
    assert "stext" not in o1[0] and o1[0]["en"] == "Hello there everyone"
    seg_cont = [{"start": 0.0, "end": 2.0, "en": "Hello there everyone",
                 "sidx": 5, "stext": full},
                {"start": 2.3, "end": 4.0, "en": "I am so happy to see you",
                 "sidx": 5, "stext": full}]         # 0.3s gap -> one line
    o2, _, _ = merge_by_script_line(seg_cont, ["Alice", "Alice"])
    assert len(o2) == 1, o2


def test_line_windows_are_word_accurate():
    # --- v2: line windows are word-accurate, split at internal pauses, and clamp
    #         out of detected silence; the gap boundaries are recorded correctly.
    sils_t = [(5.0, 6.0)]
    scr_w = [
        {"words": [[0.10, 0.40], [0.45, 0.80], [0.85, 1.20]]},       # 0.10-1.20
        {"words": [[2.00, 2.30], [6.50, 6.90]]},                     # 4.2s pause
        {"start": 7.0, "end": 8.0},                                  # no words
    ]
    w = build_line_windows(scr_w, sils_t, 20.0)
    assert abs(w[0][0] - 0.10) < 1e-6 and abs(w[0][1] - 1.20) < 1e-6, w[0]
    assert w[0][3] == 5.0, w[0]                     # next gap after this line
    assert abs(w[1][1] - 2.30) < 1e-6, w[1]         # window ends at first run
    assert abs(w[2][0] - 7.0) < 1e-6 and w[2][2] == 6.0, w[2]  # after the gap


def test_the_planner_keeps_audio_out_of_a_silent_gap():
    # --- v2: the planner keeps audio OUT of the silent gap and out of a gap start
    scr_p = [{"start": 0.0, "end": 1.0, "zh": "x", "speaker": "Alice"}]
    win_p = {0: (0.0, 1.0, 0.0, 3.0)}               # speech 0-1, gap opens at 3.0
    pp = plan_timeline(scr_p, {0: 3.0}, 30.0,
                       silences=[(3.0, 6.0)], windows=win_p)
    _, pl0, tp0 = pp[0]
    assert pl0 + 3.0 / tp0 <= 3.0 + 1e-6, pp        # never runs into the 3-6 gap
    assert pl0 >= -1e-6, pp
    win_q = {0: (6.0, 7.0, 6.0, 30.0)}              # a line right after the gap
    scr_q = [{"start": 6.0, "end": 7.0, "zh": "x", "speaker": "Alice"}]
    qq = plan_timeline(scr_q, {0: 1.0}, 30.0,
                       silences=[(3.0, 6.0)], windows=win_q)
    assert qq[0][1] >= 6.0 - 1e-6, qq               # starts at/after the gap end
    # two windowed lines around a gap never overlap and stay on their own sides
    scr_two = [{"start": 0.0, "end": 1.0, "zh": "x", "speaker": "Alice"},
               {"start": 6.0, "end": 7.0, "zh": "x", "speaker": "Bob"}]
    win_two = {0: (0.0, 1.0, 0.0, 3.0), 1: (6.0, 7.0, 6.0, 30.0)}
    tt2 = plan_timeline(scr_two, {0: 2.0, 1: 2.0}, 30.0,
                        silences=[(3.0, 6.0)], windows=win_two)
    byi = {i: (p, t) for i, p, t in tt2}
    assert byi[0][0] + 2.0 / byi[0][1] <= 3.0 + 1e-6, tt2   # line0 before the gap
    assert byi[1][0] >= 6.0 - 1e-6, tt2                     # line1 after the gap


def test_offset_extension_pushes_a_line_end_to_real_speech():
    # --- v4: reliable OFFSET extension pushes a line's end to real speech,
    #         bounded by the next line's onset, never trimming, never past it.
    assert abs(extend_end_to_speech(10.0, 11.0, [(10.0, 12.0)], 30.0) - 12.0) < 1e-6
    assert abs(extend_end_to_speech(10.0, 11.0, [(10.0, 15.0)], 12.0) - 12.0) < 1e-6  # capped at next onset
    assert abs(extend_end_to_speech(10.0, 11.5, [(10.0, 11.0)], 30.0) - 11.5) < 1e-6  # never trims inward
    assert abs(extend_end_to_speech(5.0, 6.0, [], 30.0) - 6.0) < 1e-6                 # no regions -> unchanged
    scr_ext = [{"start": 10.0, "end": 11.0, "zh": "x", "speaker": "Alice",
                "words": [[10.0, 10.4], [10.5, 11.0]]}]
    we_ext = build_line_windows(scr_ext, [], 30.0, speech=[(10.0, 12.2)])
    assert abs(we_ext[0][1] - 12.2) < 1e-6, we_ext          # end 11.0 -> real 12.2
    assert abs(build_line_windows(scr_ext, [], 30.0)[0][1] - 11.0) < 1e-6  # no VAD -> v2
    # neural VAD -> gaps: silence is the complement of the speech regions
    gg = silence_from_regions([(0.0, 4.6), (6.6, 7.6)], 7.7)
    assert gg == [(4.6, 6.6)], gg


def test_a_line_borrows_later_slack_instead_of_speeding_up():
    # --- KEY: a line is NOT sped up when a LATER line has slack it can borrow.
    #     A(onset0,clip3) B(onset2,clip1) C(onset5,clip1) with room after: nothing
    #     is compressed; B and C simply start a little late.
    scr_hb = [{"start": 0.0, "end": 1.8, "zh": "x", "speaker": "A"},
              {"start": 2.0, "end": 2.8, "zh": "x", "speaker": "B"},
              {"start": 5.0, "end": 5.8, "zh": "x", "speaker": "C"}]
    win_hb = {0: (0.0, 1.8, 0.0, 8.0),
              1: (2.0, 2.8, 0.0, 8.0),
              2: (5.0, 5.8, 0.0, 8.0)}
    hb = {i: (p, t) for i, p, t in
          plan_timeline(scr_hb, {0: 3.0, 1: 1.0, 2: 1.0}, 10.0,
                        silences=[(8.0, 10.0)], windows=win_hb)}
    for i in (0, 1, 2):
        assert hb[i][1] <= 1.0 + 1e-6, ("should NOT speed up", i, hb[i])
    assert hb[0][0] < 1e-6, hb                     # A on its onset


def test_planner_invariants_hold_across_random_scenes():
    # --- randomized invariant harness: whatever the scene, the HARD guarantees
    #     always hold at once -- no overlap, no clip starts before its onset, and
    #     tempo is in [1, MAX]. (Gaps are intentionally NOT preserved now: a clip
    #     may play on into empty time, so that is not asserted.)
    import random as _rnd
    _rnd.seed(1234)
    for _ in range(500):
        n = _rnd.randint(1, 10)
        scr, dur = [], {}
        t = _rnd.uniform(0, 1.0)
        for i in range(n):
            o0 = t
            we0 = o0 + _rnd.uniform(0.4, 2.5)
            scr.append({"start": round(o0, 3), "end": round(we0, 3),
                        "zh": "x", "speaker": "A"})
            dur[i] = round(_rnd.uniform(0.3, 4.0), 3)
            t = we0 + _rnd.uniform(0.05, 3.0)      # empty time before the next line
        total = t + _rnd.uniform(0.5, 3.0)
        plan_r = plan_timeline(scr, dur, total)
        srt = sorted(plan_r, key=lambda x: x[1])
        pend = None
        for idx, place, tempo in srt:
            assert 1.0 - 1e-9 <= tempo <= MAX_SPEEDUP + 1e-6, (tempo,)
            assert place >= scr[idx]["start"] - 1e-6, ("started early", idx, place)
            if pend is not None:
                assert place >= pend - 1e-6, ("overlap", place, pend)
            pend = place + dur[idx] / tempo
