"""Config resolution."""

import json
from pathlib import Path

from peiyin.config import CONFIG_PATH, DEFAULT_OUT_DIR, current_out_dir, load_config


def test_the_output_folder_resolver_honours_an_override():
    # --- output folder resolver: config override, blank -> default, both exist
    cfgo = load_config()
    saved_out = cfgo.get("output_dir")
    import tempfile as _tf
    custom = Path(_tf.mkdtemp()) / "My Dubs"
    cfgo["output_dir"] = str(custom)
    CONFIG_PATH.write_text(json.dumps(cfgo, ensure_ascii=False), encoding="utf-8")
    assert current_out_dir() == custom and custom.exists()
    cfgo = load_config()
    cfgo["output_dir"] = ""
    CONFIG_PATH.write_text(json.dumps(cfgo, ensure_ascii=False), encoding="utf-8")
    assert current_out_dir() == DEFAULT_OUT_DIR
    cfgo = load_config()
    if saved_out is None:
        cfgo.pop("output_dir", None)
    else:
        cfgo["output_dir"] = saved_out
    CONFIG_PATH.write_text(json.dumps(cfgo, ensure_ascii=False), encoding="utf-8")
