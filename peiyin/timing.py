"""The timing planner — pure functions, no I/O.

Given line onsets and ends, synthesised clip durations, and the episode's
silence regions, decide where each dubbed line is placed and how much (if at
all) it is sped up.

The rules, unchanged from the original:
  * a line starts on its onset and never earlier;
  * it plays at natural speed for as long as it needs, running into free
    silence rather than being cut short;
  * it may drift a bounded amount late to make room, and drift drains at the
    next real gap;
  * speed-up is the last resort, applied only as much as needed;
  * no line ever overlaps another line's audio.

Every function here takes and returns plain data, so the whole planner is
testable without audio files or network access.
"""

MIN_GAP = 0.12          # guaranteed air between consecutive lines (seconds)
SPLIT_GAP = 2.0         # a pause longer than this splits one speech into pieces
DRIFT = 1.5             # legacy slack (kept for window bounds / tests)
MAX_SPEEDUP = 1.6       # a line may be sped up this much, but ONLY as a last resort
MAX_LAG = 4.0           # a line may play this long past its own words / start this


MIN_SILENCE = 0.35      # a quiet stretch must last this long to count as a gap
SIL_CLOSE = 0.18        # quiet dips shorter than this are bridged (within-word)
SIL_MIN_SPEECH = 0.12   # speech blips shorter than this are ignored
SIL_THRESH_FRAC = 0.20  # gap if frame power-dB below floor + this*(peak-floor)


def silence_from_regions(regions, total_len, min_sil=MIN_SILENCE):
    """The quiet gaps between speech regions (scene changes / dead air)."""
    gaps, t = [], 0.0
    for s, e in sorted(regions):
        if s - t >= min_sil:
            gaps.append((round(t, 3), round(s, 3)))
        t = max(t, e)
    if total_len - t >= min_sil:
        gaps.append((round(t, 3), round(total_len, 3)))
    return gaps


def extend_end_to_speech(o0, we0, regions, next_onset):
    """RELIABLE offset fix: extend a line's end to where speech truly stops.

    Whisper's last-word time usually lands early; the planner then thinks the
    line is shorter than it is and speeds it up. We push the end out to the end
    of the speech region the line sits in (from the neural VAD), but never past
    the next line's onset (so one line can't swallow the next) and never inward
    (a region that the VAD split can only give MORE room, never less).
    """
    end = we0
    for s, e in regions:
        if e > o0 and s < we0 + 0.4:        # a region overlapping / meeting this line
            end = max(end, e)
    end = min(end, next_onset)               # never cross into the next line
    return max(end, we0)


def build_line_windows(script, silences, total_len, speech=None):
    """For every line, compute its REAL speech window and the silence boundaries
    around it. Returns {idx: (onset, word_end, sil_before_end, sil_after_start)}.

      onset / word_end  -- from the line's word timestamps (falling back to the
                           segment start/end if a line has no words). If a large
                           pause sits *inside* the words, the window ends at the
                           first speech run so a line can't span a gap. If neural
                           VAD `speech` regions are supplied, word_end is EXTENDED
                           to where speech truly stops (Whisper cuts ends short),
                           bounded by the next line's onset -- never trimmed, never
                           past a neighbour. The onset is left on the first word.
                           The window is finally clamped so it never crosses a
                           detected silence.
      sil_before_end    -- end of the quiet gap just before the line (a line may
                           not start/drift earlier than this).
      sil_after_start   -- start of the quiet gap just after the line (a line's
                           audio may not run/drift past this).
    """
    sil = sorted(silences or [])
    regions = sorted(speech or [])

    # pass 1: raw word windows (also give us each neighbour's onset)
    raw = []
    for ln in script:
        words = [w for w in (ln.get("words") or [])
                 if isinstance(w, (list, tuple)) and len(w) >= 2 and w[1] > w[0]]
        if words:
            o0 = float(words[0][0])
            we0 = float(words[-1][1])
            for a, b in zip(words, words[1:]):    # split at a big internal pause
                if b[0] - a[1] > SPLIT_GAP:
                    we0 = float(a[1])
                    break
        else:
            o0 = float(ln.get("start", 0.0))
            we0 = float(ln.get("end", o0 + 0.4))
        raw.append((o0, max(we0, o0 + 0.3)))

    win = {}
    for i, (onset, word_end) in enumerate(raw):
        # reliable OFFSET fix: push the end out to where speech really stops,
        # bounded by the next line's onset so nothing swallows a neighbour.
        if regions:
            next_onset = raw[i + 1][0] if i + 1 < len(raw) else float(total_len)
            word_end = extend_end_to_speech(onset, word_end, regions, next_onset)

        # keep the window out of any detected silence (unchanged from v2)
        for s, e in sil:
            if s - 1e-6 <= onset < e:          # onset sits inside a gap -> nudge
                onset = e
            if s < word_end <= e + 1e-6:        # end sits inside a gap -> pull in
                word_end = s
        for s, e in sil:                        # clamp to the first gap after onset
            if onset < s < word_end:
                word_end = s
                break
        word_end = max(word_end, onset + 0.3)
        sil_before_end = 0.0
        for s, e in sil:
            if e <= onset + 1e-6:
                sil_before_end = e
            else:
                break
        sil_after_start = float(total_len)
        for s, e in sil:
            if s >= word_end - 1e-6:
                sil_after_start = s
                break
        win[i] = (round(onset, 4), round(word_end, 4),
                  round(sil_before_end, 4), round(sil_after_start, 4))
    return win


def plan_timeline(script, durations, total_len, silences=None, windows=None):
    """Place each Chinese line and decide its speed. Returns [(idx, start, tempo)].

    The rule (v5), matching how dubbing should actually feel:
      * A line STARTS on its onset (never early). It then plays at NORMAL speed
        for as long as it needs, running freely into whatever empty time follows
        it -- silence, a beat, a scene gap -- because nobody is talking there.
      * The ONLY thing that stops a line is the NEXT line's audio. A line may run
        up to MIN_GAP before the next line starts. If the next line's audio is
        soon, the next line is first allowed to itself start a little late (up to
        MAX_LAG past its onset) to make room -- drift before speed-up.
      * Speed-up (<= MAX_SPEEDUP) is the LAST resort, used only when even that
        isn't enough room.

    So a line is never cut short just because its words ended or a silence began;
    silences are NOT preserved -- the Chinese simply keeps playing over them. Two
    clips never overlap, and because each line may only be pushed MAX_LAG past its
    own onset, a backlog drains at every real gap (the next onset is far, so the
    next line lands on its onset and the lag resets).

    `windows` (from build_line_windows) supplies each line's word-accurate onset;
    with windows=None the segment start is used. Silences/word-ends are no longer
    needed to place -- the next line's onset already says where the free time ends.
    """
    def _onset(i):
        if windows and i in windows:
            return float(windows[i][0])
        return float(script[i]["start"])

    order = [i for i in sorted(durations) if durations[i] > 0]
    order.sort(key=lambda i: (_onset(i), i))
    m = len(order)
    if m == 0:
        return []

    o = [_onset(i) for i in order]
    d = [float(durations[i]) for i in order]

    # ---- backward pass: the latest each line's AUDIO may end (its deadline).
    # A line may run into the empty time after it, but must leave the next line
    # its slot. The next line is allowed to start up to MAX_LAG past its onset
    # (that's the room this line can borrow); beyond that, this line speeds up.
    # The last line has the whole rest of the file to breathe into.
    deadline = [0.0] * m
    for k in range(m - 1, -1, -1):
        if k == m - 1:
            deadline[k] = float(total_len)
        else:
            # next line may sit as late as o[k+1]+MAX_LAG, or wherever its own
            # deadline (minus its shortest possible length) forces it -- whichever
            # is EARLIER is the wall this line must end before.
            next_latest_start = min(o[k + 1] + MAX_LAG,
                                    deadline[k + 1] - d[k + 1] / MAX_SPEEDUP)
            deadline[k] = next_latest_start - MIN_GAP

    # ---- forward pass: start on the onset (never early); run at normal speed;
    # speed up only if we'd otherwise cross into the next line's audio.
    plan = []
    prev_end = None
    for k in range(m):
        floor = (prev_end + MIN_GAP) if prev_end is not None else 0.0
        place = max(o[k], floor)                    # ON the onset, or wait for prev
        room = deadline[k] - place
        if room >= d[k] - 1e-9:
            tempo = 1.0                             # plenty of empty time: let it play
        else:
            tempo = min(d[k] / max(room, 0.3), MAX_SPEEDUP)   # last resort
        tempo = round(min(tempo, MAX_SPEEDUP), 4)
        fitted = d[k] / tempo
        plan.append((order[k], round(place, 4), tempo))
        prev_end = place + fitted
    return plan
