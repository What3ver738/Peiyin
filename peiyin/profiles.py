"""Show profiles — loading, validation and the active-profile accessors.

A profile is a TOML file describing one show: its characters and their aliases,
the transcript source the user configured, the target language, and optional
intro/theme handling. Profiles are what make Peiyin general rather than tied
to any one show.

Peiyin ships no transcripts and no voice ids. A profile names characters and
says where *your* transcripts live; it never carries content.

Profiles are searched for in two places, user files winning on a name clash:

  * ``<config dir>/profiles/`` -- profiles the user created
  * ``profiles/`` next to the package -- the examples shipped with Peiyin

The rest of the app reads the active profile through the accessors at the
bottom of this module (``main_cast()``, ``labels()``, ...) rather than through
module-level constants, so switching profiles in the UI takes effect
immediately.
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

try:                                     # Python 3.11+
    import tomllib
except ModuleNotFoundError:              # pragma: no cover - 3.10 fallback
    import tomli as tomllib

from .config import APP_DIR, load_config, save_config

#: Where user-created profiles live.
USER_PROFILE_DIR = APP_DIR / "profiles"
# Included as package data in both wheels and source distributions.
BUNDLED_PROFILE_DIR = Path(__file__).resolve().parent / "data" / "profiles"

GENERIC_SPEAKERS = ["OtherMale", "OtherFemale"]
SKIP_LABEL = "Skip"

#: Transcript formats Peiyin can read. `episodes.PARSERS` implements exactly
#: these; adding a format means writing one function and naming it in both.
KNOWN_FORMATS = ("name_colon", "srt_with_speakers", "html")


class ProfileError(ValueError):
    """A profile file is missing something, or says something contradictory."""


@dataclass(frozen=True)
class Character:
    name: str
    aliases: tuple = ()
    gender: str = ""          # "M", "F" or "" (unknown)
    main: bool = False


@dataclass(frozen=True)
class Transcripts:
    """Where this profile's transcripts come from. Empty means no-transcript mode."""

    local_folder: str = ""
    url_list: tuple = ()
    url_base: tuple = ()
    url_suffix: str = ""
    fmt: str = "name_colon"
    revision: str = ""
    episodes_per_season: tuple = ()
    multi_part_pages: tuple = ()
    covered_by_multi: tuple = ()

    @property
    def configured(self):
        return bool(self.local_folder or self.url_list or self.url_base)


@dataclass(frozen=True)
class Theme:
    skip_intro_seconds: float = 0.0
    theme_words: tuple = ()


@dataclass(frozen=True)
class Profile:
    name: str
    target_language: str = "zh-CN"
    characters: tuple = ()
    transcripts: Transcripts = field(default_factory=Transcripts)
    theme: Theme = field(default_factory=Theme)
    path: str = ""

    # ---- derived views the rest of the app uses ----------------------------
    @property
    def main_cast(self):
        """Main characters, in profile order."""
        return [c.name for c in self.characters if c.main]

    @property
    def speakers(self):
        """Every voiced label: the main cast plus the generic guest buckets."""
        return self.main_cast + list(GENERIC_SPEAKERS)

    @property
    def labels(self):
        """Every label the app may assign, including Skip."""
        return self.speakers + [SKIP_LABEL]

    @property
    def male_roles(self):
        return {c.name for c in self.characters
                if c.main and c.gender.upper().startswith("M")} | {"OtherMale"}

    @property
    def aliases(self):
        """Single-word label -> character name (lowercased keys).

        A character's own name is always an alias of itself.
        """
        out = {}
        for c in self.characters:
            if not c.main:
                continue
            out[c.name.lower()] = c.name
            for a in c.aliases:
                a = a.strip().lower()
                if a and len(a.split()) == 1:
                    out[a] = c.name
        return out

    @property
    def full_names(self):
        """Multi-word label -> character name (lowercased keys)."""
        out = {}
        for c in self.characters:
            if not c.main:
                continue
            for a in c.aliases:
                a = " ".join(a.strip().lower().split())
                if a and len(a.split()) > 1:
                    out[a] = c.name
        return out

    @property
    def has_transcripts(self):
        return self.transcripts.configured


def _as_tuple(v):
    if v is None:
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return (v,)


def parse_profile(data, path=""):
    """Turn parsed TOML into a Profile, or raise ProfileError."""
    if not isinstance(data, dict):
        raise ProfileError("a profile must be a TOML table")
    name = str(data.get("name") or "").strip()
    if not name:
        raise ProfileError("profile needs a non-empty `name`")

    language = str(data.get("target_language") or "zh-CN")
    if language != "zh-CN":
        raise ProfileError("Peiyin currently supports Mandarin output only; use zh-CN.")
    chars = []
    seen = set()
    for raw in data.get("characters") or []:
        cname = str(raw.get("name") or "").strip()
        if not cname:
            raise ProfileError(f"{name}: every [[characters]] entry needs a `name`")
        if cname.lower() in seen:
            raise ProfileError(f"{name}: duplicate character {cname!r}")
        if cname in GENERIC_SPEAKERS or cname == SKIP_LABEL:
            raise ProfileError(
                f"{name}: {cname!r} is reserved -- Peiyin uses it internally")
        seen.add(cname.lower())
        gender = str(raw.get("gender") or "").strip().upper()[:1]
        if gender not in ("M", "F", ""):
            raise ProfileError(
                f"{name}: character {cname!r} has gender {raw.get('gender')!r}; "
                "use \"M\", \"F\" or leave it out")
        chars.append(Character(name=cname,
                               aliases=_as_tuple(raw.get("aliases")),
                               gender=gender,
                               main=bool(raw.get("main", True))))
    if not any(c.main for c in chars):
        raise ProfileError(
            f"{name}: no main characters. Mark at least one [[characters]] "
            "entry with main = true.")

    t = data.get("transcripts") or {}
    fmt = str(t.get("format") or "name_colon").strip()
    if fmt not in KNOWN_FORMATS:
        raise ProfileError(
            f"{name}: unknown transcript format {fmt!r}. "
            f"Known formats: {', '.join(sorted(KNOWN_FORMATS))}.")
    transcripts = Transcripts(
        local_folder=str(t.get("local_folder") or "").strip(),
        url_list=_as_tuple(t.get("url_list")),
        url_base=_as_tuple(t.get("url_base")),
        url_suffix=str(t.get("url_suffix") or "").strip(),
        fmt=fmt,
        revision=str(t.get("revision") or ""),
        episodes_per_season=_as_tuple(t.get("episodes_per_season")),
        multi_part_pages=_as_tuple(t.get("multi_part_pages")),
        covered_by_multi=_as_tuple(t.get("covered_by_multi")),
    )

    th = data.get("theme") or {}
    try:
        skip = float(th.get("skip_intro_seconds") or 0.0)
    except (TypeError, ValueError):
        raise ProfileError(f"{name}: skip_intro_seconds must be a number")
    if skip < 0:
        raise ProfileError(f"{name}: skip_intro_seconds cannot be negative")
    theme = Theme(skip_intro_seconds=skip,
                  theme_words=tuple(str(w).lower() for w in _as_tuple(th.get("theme_words"))))

    return Profile(name=name,
                   target_language=str(data.get("target_language") or "zh-CN"),
                   characters=tuple(chars),
                   transcripts=transcripts,
                   theme=theme,
                   path=str(path))


def load_profile(path):
    """Read and validate one profile file."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProfileError(f"no profile at {path}")
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"{path.name} is not valid TOML: {e}")
    return parse_profile(data, path=str(path))


def profile_paths():
    """{profile id: path}. A user profile shadows a bundled one of the same id."""
    out = {}
    for d in (BUNDLED_PROFILE_DIR, USER_PROFILE_DIR):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.toml")):
            if f.stem.upper() == "TEMPLATE":
                continue
            out[f.stem] = f
    return out


def list_profiles():
    """[(id, display name)] for every readable profile."""
    out = []
    for pid, path in sorted(profile_paths().items()):
        try:
            out.append((pid, load_profile(path).name))
        except ProfileError:
            continue
    return out


def active_profile_id():
    return (load_config().get("profile") or "").strip()


def set_active_profile(profile_id):
    """Switch profiles. Returns the newly active Profile."""
    if profile_id not in profile_paths():
        raise ProfileError(f"no profile called {profile_id!r}")
    save_config(profile=profile_id)
    active.cache_clear()
    return active()


def create_profile(profile_id, show_name=""):
    """Write a new profile from TEMPLATE.toml into the user profile dir."""
    profile_id = re.sub(r"[^a-z0-9_-]+", "_", (profile_id or "").strip().lower()).strip("_")
    if not profile_id:
        raise ProfileError("give the profile a name")
    dest = USER_PROFILE_DIR / f"{profile_id}.toml"
    if dest.exists():
        raise ProfileError(f"{dest.name} already exists")
    template = BUNDLED_PROFILE_DIR / "TEMPLATE.toml"
    if not template.exists():
        raise ProfileError("TEMPLATE.toml is missing from the install")
    USER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, dest)
    if show_name:
        text = dest.read_text(encoding="utf-8")
        text = re.sub(r'(?m)^name\s*=.*$',
                      lambda _: 'name = ' + json.dumps(show_name, ensure_ascii=False), text, count=1)
        dest.write_text(text, encoding="utf-8")
    return dest


# --------------------------------------------------------------- active state

def _fallback_profile():
    """Used when nothing is configured yet, so the app still starts."""
    return Profile(name="(no profile selected)",
                   characters=(Character(name="Speaker", gender="", main=True),))


def active():
    """The profile in use. Cached; call `active.cache_clear()` after a change."""
    if getattr(active, "_cached", None) is not None:
        return active._cached
    paths = profile_paths()
    pid = active_profile_id()
    chosen = paths.get(pid)
    if chosen is None and len(paths) == 1:
        chosen = next(iter(paths.values()))
    prof = _fallback_profile()
    if chosen is not None:
        try:
            prof = load_profile(chosen)
        except ProfileError:
            raise
    active._cached = prof
    return prof


active._cached = None
active.cache_clear = lambda: setattr(active, "_cached", None)


# Accessors the rest of the app uses instead of module-level constants, so a
# profile switch takes effect without restarting.
def main_cast():
    return active().main_cast


def speakers():
    return active().speakers


def labels():
    return active().labels


def male_roles():
    return active().male_roles


def target_language():
    return active().target_language
