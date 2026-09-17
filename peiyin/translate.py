"""Translation into the target language.

One LLM pass over the whole script, so the model knows who speaks each line,
with optional short delivery cues like `[excited]` for TTS providers that
support them.
"""

import json

from .config import DIAG
from .llm import llm_json
from .tts import clean_emotion

TRANSLATE_SYSTEM = (
    "You are the Mandarin dubbing writer for a TV show. Translate "
    "each line into natural, casual spoken Mandarin (Simplified characters) "
    "matched to the speaker's voice and personality. Keep jokes funny in "
    "Chinese, adapting where a literal translation would kill them. "
    "IMPORTANT for dubbing: each line has a 'max' number of characters -- the "
    "spoken Chinese should be at most that many characters so it fits the time "
    "the actor's mouth is moving. Meet the limit by saying it more concisely "
    "(drop filler, use shorter synonyms), NEVER by dropping meaning. Going a "
    "few characters over is fine; being far over makes the dub sound rushed. "
    "The voice engine can act out a delivery cue. For each line you MAY add an "
    "'emo' field: ONE short English delivery cue in square brackets that is "
    "placed before the speech, e.g. [excited], [angry], [shouting], "
    "[whispering], [sad], [crying], [sarcastic], [nervous], [laughing]. Only "
    "add it when the line clearly calls for it -- a shout, a sob, obvious "
    "excitement or anger. Leave 'emo' empty for ordinary conversational lines, "
    "which is MOST lines. Do NOT put the cue inside the 'zh' text. "
    'Respond with JSON only: '
    '{"lines":[{"i":<index>,"zh":"<Chinese>","emo":"<cue or empty>"}]} with '
    "one entry per input line."
)


CHARS_PER_SEC = 4.6
MIN_CHAR_BUDGET = 4


def translate_llm(script, llm_key, workdir, progress=None):
    def missing():
        return [i for i, ln in enumerate(script)
                if ln["speaker"] != "Skip" and not ln.get("zh")]

    def do_batch(idxs):
        payload = []
        for i in idxs:
            span = max(script[i].get("end", 0) - script[i].get("start", 0), 0.6)
            budget = max(int(span * CHARS_PER_SEC), MIN_CHAR_BUDGET)
            payload.append({"i": i, "speaker": script[i]["speaker"],
                            "en": script[i]["en"], "max": budget})
        ctx_lo = max(0, idxs[0] - 6)
        ctx = "\n".join(f"{script[j]['speaker']}: {script[j]['en']}"
                        for j in range(ctx_lo, idxs[0]))
        user = ""
        if ctx:
            user += "Previous lines for context (do not translate):\n" + ctx + "\n\n"
        user += ("Translate EVERY line below; return exactly one entry per input "
                 "index, never skip one.\n"
                 + json.dumps(payload, ensure_ascii=False))
        try:
            data = llm_json(llm_key, TRANSLATE_SYSTEM, user)
        except Exception:  # noqa: BLE001
            return                    # leave these for the next (smaller) pass
        for item in data.get("lines", []):
            i = item.get("i")
            if (isinstance(i, int) and 0 <= i < len(script)
                    and str(item.get("zh", "")).strip()):
                script[i]["zh"] = str(item.get("zh", "")).strip()
                script[i]["emo"] = clean_emotion(item.get("emo", ""))

    total = len(missing()) or 1
    # Shrinking batches: a big batch is efficient, but the model sometimes omits
    # lines from a long JSON list. Re-run only the still-missing lines with
    # smaller and smaller batches (ending one-at-a-time) so nothing is dropped.
    for batch in (30, 12, 4, 1):
        rem = missing()
        if not rem:
            break
        for b in range(0, len(rem), batch):
            do_batch(rem[b:b + batch])
            if progress:
                done = total - len(missing())
                progress(min(done / total, 1.0),
                         desc=f"Translating... {done}/{total}")
    left = missing()
    if left:
        DIAG.append(f"{len(left)} line(s) would not translate even one-by-one; "
                    f"their original English is kept in the subtitles so they "
                    f"are not lost.")
    return script
