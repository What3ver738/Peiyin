"""Settings, credentials and filesystem locations.

Holds the user config directory, `.env` loading, API-key resolution
(environment variable first, then the stored config file) and the service
endpoint constants. Nothing here is show-specific — see `peiyin.profiles`.
"""

import json
import os
import tempfile
import threading
from pathlib import Path

import imageio_ffmpeg

from . import __version__

# Keep localhost out of any proxy the user has configured, so the local web
# interface is always reachable.
for _var in ("NO_PROXY", "no_proxy"):
    _parts = [x.strip() for x in os.environ.get(_var, "").split(",") if x.strip()]
    for _h in ("127.0.0.1", "localhost", "0.0.0.0"):
        if _h not in _parts:
            _parts.append(_h)
    os.environ[_var] = ",".join(_parts)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


SR = 44100



PRODUCT_VERSION = __version__
# PEIYIN_HOME relocates everything Peiyin stores (config, caches, work
# folders, default output). Tests set it so they never touch a real install.
_HOME = Path(os.environ.get("PEIYIN_HOME") or Path.home())
APP_DIR = _HOME / ".peiyin"
APP_VERSION = 17               # content-aware caches; old work stays untouched


WORK_ROOT = APP_DIR / f"work_v{APP_VERSION}"


DEFAULT_OUT_DIR = _HOME / "Peiyin Output"
CONFIG_PATH = APP_DIR / "config.json"
for _d in (APP_DIR, WORK_ROOT, DEFAULT_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True, mode=0o700)

if os.name != "nt":
    APP_DIR.chmod(0o700)

DIAG = []

# The cast is not a constant: it comes from the active show profile. Use
# peiyin.profiles.main_cast() / speakers() / labels() / male_roles().


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"


def setting(name, default):
    """A user-overridable setting, from the config file."""
    v = load_config().get(name)
    return v if v not in (None, "") else default


def fish_model():
    return setting("fish_model", FISH_TTS_MODEL)


def llm_model():
    """Empty selects the documented provider default; never probes paid models."""
    return setting("llm_model", "")


FISH_TTS_URL = "https://api.fish.audio/v1/tts"
FISH_MODEL_URL = "https://api.fish.audio/model"     # only used to check the key
FISH_TTS_MODEL = "s2.1-pro"                         # paid flagship; [bracket] emotion


def load_dotenv(path=None):
    """Read a .env file into os.environ without overwriting a real env var.

    Looked for next to the project by default. Missing file is fine -- keys can
    equally be typed into the app or exported in the shell.
    """
    path = Path(path) if path else (Path.cwd() / ".env")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    found = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            found[k] = v
            os.environ.setdefault(k, v)
    return found


load_dotenv()


#: Config key -> the environment variable that overrides it. An environment
#: variable always wins, so a CI run or a shared machine need never write a key
#: to disk.
ENV_KEYS = {
    "fish_key": "FISH_API_KEY",
    "openai_key": "OPENAI_API_KEY",
    "gemini_key": "GEMINI_API_KEY",
}


def load_config():
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot read Peiyin settings; restore or repair config.json.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Peiyin config.json must contain an object.")
    return data


def api_key(which):
    """Resolve one credential: environment variable first, then stored config.

    `which` is "fish", "openai" or "gemini". Returns "" when unset.
    """
    name = f"{which}_key"
    env = os.environ.get(ENV_KEYS.get(name, ""), "")
    if env.strip():
        return env.strip()
    cfg = load_config()
    if name == "fish_key":
        return (cfg.get("fish_key") or "").strip()
    # One box in the UI takes either an OpenAI or a Gemini key.
    stored = (cfg.get("llm_key") or "").strip()
    if which == "openai":
        return stored if stored.startswith("sk-") else ""
    return "" if stored.startswith("sk-") else stored


def llm_key():
    """Whichever translation/casting key is available: OpenAI, else Gemini."""
    return api_key("openai") or api_key("gemini")


def key_source(which):
    """Where a key came from, for the settings UI: env var, config, or unset."""
    name = f"{which}_key"
    if os.environ.get(ENV_KEYS.get(name, ""), "").strip():
        return "environment"
    return "config file" if api_key(which) else "not set"


_CONFIG_LOCK = threading.RLock()


def save_config(**kw):
    """Atomically merge settings. Blank strings keep keys; None removes them.

    mkstemp creates a private file before any secret is written. Replacement
    preserves the previous config on failure. Windows uses the user's folder
    ACLs; POSIX mode bits are not a Windows access-control guarantee.
    """
    with _CONFIG_LOCK:
        cfg = load_config()
        for k, v in kw.items():
            if v is None:
                cfg.pop(k, None)
            elif v != "":
                cfg[k] = v
        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(prefix=".config-", dir=CONFIG_PATH.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, CONFIG_PATH)
        except OSError as exc:
            raise RuntimeError("Could not save Peiyin settings. Check folder permissions and disk space.") from exc
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)


def resolve_keys(llm="", fish=""):
    """Environment wins over transient UI input, then saved credentials."""
    env_llm = (os.environ.get("OPENAI_API_KEY", "").strip()
               or os.environ.get("GEMINI_API_KEY", "").strip())
    env_fish = os.environ.get("FISH_API_KEY", "").strip()
    return (env_llm or (llm or "").strip() or llm_key(),
            env_fish or (fish or "").strip() or api_key("fish"))


def current_out_dir():
    """The folder finished episodes are written to. Defaults to a visible
    ~/Peiyin Output folder, or whatever folder the user set in the app.
    Always exists."""
    raw = (load_config().get("output_dir") or "").strip()
    try:
        d = DEFAULT_OUT_DIR if not raw else Path(raw).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_OUT_DIR


SCRIPT_DIR = APP_DIR / "scripts"


SPK_MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                 "speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx")
SPK_MODEL_SHA256 = "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef"
SPK_MODEL_PATH = APP_DIR / "models" / "speaker_embedding.onnx"
