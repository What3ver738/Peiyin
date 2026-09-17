"""Translation: completeness and cache keying."""

import json
import re
import tempfile
from pathlib import Path

from peiyin.config import DIAG
from peiyin.pipeline import _reuse_translations
from peiyin.translate import translate_llm


def test_translation_never_drops_a_line():
    # --- translation never drops a line: a model that omits lines from a big
    # batch must still get them all via the shrinking-batch retries.
    from peiyin import translate as _translate
    _real_llm = _translate.llm_json

    def _fake_llm(key, system, user):
        m = re.search(r"\[.*\]", user, re.S)
        items = json.loads(m.group(0)) if m else []
        out = []
        for it in items:
            if len(items) > 3 and it["i"] % 5 == 0:      # drop some in big batches
                continue
            out.append({"i": it["i"], "zh": "\u8bd1" + str(it["i"]), "emo": ""})
        return {"lines": out}

    _translate.llm_json = _fake_llm
    tscript = [{"start": k * 1.0, "end": k * 1.0 + 0.8, "en": f"line {k}",
                "speaker": "Alice", "zh": "", "emo": ""} for k in range(20)]
    translate_llm(tscript, "sk-test", Path(tempfile.mkdtemp()))
    _translate.llm_json = _real_llm
    assert all(ln["zh"] for ln in tscript), \
        [k for k, ln in enumerate(tscript) if not ln["zh"]]


def test_reused_translations_keep_their_own_wording(tmp):
    # --- REGRESSION: reusing translations must not flatten repeated lines
    wd = tmp / "reuse"
    wd.mkdir(exist_ok=True)
    (wd / "script8_llm.json").write_text(json.dumps([
        {"en": "What?", "zh": "\u4ec0\u4e48\uff1f", "emo": "angry", "sidx": 3},
        {"en": "What?", "zh": "\u5565\uff1f", "emo": "confused", "sidx": 9},
        {"en": "I am fine", "zh": "\u6211\u5f88\u597d", "emo": "", "sidx": 5},
    ], ensure_ascii=False), encoding="utf-8")
    fresh = [{"en": "What?", "zh": "", "emo": "", "sidx": 3, "speaker": "Alice"},
             {"en": "What?", "zh": "", "emo": "", "sidx": 9, "speaker": "Erin"},
             {"en": "I am fine", "zh": "", "emo": "", "sidx": 5, "speaker": "Amy"},
             {"en": "What?", "zh": "", "emo": "", "sidx": 42, "speaker": "New"}]
    _reuse_translations(fresh, wd, "llm")
    assert fresh[0]["emo"] == "angry" and fresh[1]["emo"] == "confused", fresh
    assert fresh[0]["zh"] != fresh[1]["zh"], "repeated lines were flattened!"
    assert fresh[2]["zh"] == "\u6211\u5f88\u597d", fresh          # unambiguous en
    assert fresh[3]["zh"] == "", fresh          # ambiguous -> left for translator
    DIAG.clear()
