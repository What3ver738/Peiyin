"""Text-to-speech, behind a provider interface.

Peiyin v1 ships one provider (Fish Audio). Everything the rest of the app
needs from a TTS backend is declared in `TTSProvider` below, so adding another
backend means implementing this interface and registering it -- no changes
anywhere else.

Also holds voice casting: which voice id is used for which character, the
guest-voice pool, and the gender/pitch screening applied to guest voices.

**No default voice ids ship with this project.** Users choose their own, and
are responsible for only using voices they have the right to use.
"""

import hashlib
import re
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from .config import (
    DIAG,
    FISH_MODEL_URL,
    FISH_TTS_URL,
    fish_model,
    load_config,
    save_config,
)
from .llm import llm_json_model, pick_openai_model
from .profiles import speakers
from .transcribe import estimate_median_f0
from .util import to_std_wav


def parse_guest_pool(text):
    """Parse the guest-voice box. One voice per line, gender then id, e.g.
        M 1a2b3c...          F 女大学生 913e2e...
        male: 1a2b3c...      女 5c353f...
    The gender is taken from the first token (m/male/男 or f/female/女) and the
    id is the long hex string on the line (any label in between is ignored).
    Returns [(\"MALE\"|\"FEMALE\", reference_id), ...]."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if line[0] in "女" or low.startswith(("f", "female")):
            g = "FEMALE"
        elif line[0] in "男" or low.startswith(("m", "male")):
            g = "MALE"
        else:
            continue                                   # no gender -> skip
        hexm = re.search(r"[0-9a-fA-F]{24,}", line)
        if hexm:
            out.append((g, hexm.group(0)))
    return out


def load_fish_guest_pool():
    return parse_guest_pool(load_config().get("fish_guest_pool_text", ""))


FEMALE_HINTS = ["女", "妹", "姐", "娘", "少女", "女生", "女声", "女士", "小姐",
                "阿姨", "奶奶", "姑娘", "女性", "御姐", "萝莉", "女孩", "女神",
                "妈妈", "母亲", "太太", "female", "woman", "girl", "lady"]
MALE_HINTS = ["男", "哥", "弟弟", "少年", "男生", "男声", "先生", "大叔", "叔叔",
              "爷", "男性", "帅哥", "男孩", "爸爸", "父亲", "大哥", "male",
              "boy", "gentleman", "guy"]


def _has_cjk(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in (s or ""))


VOICE_DENYLIST = [
    "sexy", "seductive", "seduction", "sensual", "erotic", "nsfw", "moan",
    "moaning", "breathy", "asmr", "whisper", "kiss", "flirt", "flirty",
    "girlfriend", "boyfriend", "sultry", "intimate", "bedroom", "18+",
    "性感", "娇喘", "呻吟", "诱惑", "魅惑", "耳语", "助眠", "撩", "暧昧", "挑逗",
    "骚", "湿", "喘息", "喘", "小奶音", "哄睡", "女友", "男友", "床", "情欲",
    "娇滴滴", "嗲",
]


def _is_bad_voice(text):
    """True if a voice's metadata marks it as sexy/breathy/ASMR/etc."""
    t = (text or "").lower()
    return any(bad in t for bad in VOICE_DENYLIST)


def _guess_voice_gender(text):
    """Guess a Fish voice's gender from its title/description. '' if unclear."""
    t = (text or "").lower()
    if any(h in t for h in FEMALE_HINTS):
        return "FEMALE"
    if any(h in t for h in MALE_HINTS):
        return "MALE"
    return ""


def fetch_fish_public_pool(fish_key, cap=20, pages=6):
    """Fetch popular PUBLIC Mandarin voices from Fish, split by the gender in
    each voice's title/description. Excludes voices already cast. Returns
    (males, females) lists of reference_ids. Network failures return whatever
    was gathered so far (an empty result is simply not cached, so it retries)."""
    reserved = set(load_voice_cast().values())
    males, females = [], []
    headers = {"Authorization": "Bearer " + fish_key}

    def is_chinese(it):
        langs = [str(x).lower() for x in (it.get("languages") or [])]
        return (any(lang.startswith("zh") for lang in langs)
                or _has_cjk(it.get("title", "")))

    def harvest(params):
        for page in range(1, pages + 1):
            try:
                with httpx.Client(timeout=60, trust_env=True) as c:
                    r = c.get(FISH_MODEL_URL, headers=headers,
                              params={**params, "page_size": 50,
                                      "page_number": page})
                if r.status_code != 200:
                    break
                data = r.json()
            except Exception:  # noqa: BLE001
                break
            items = data.get("items", []) if isinstance(data, dict) else []
            if not items:
                break
            for it in items:
                vid = it.get("_id")
                if (not vid or vid in reserved
                        or it.get("visibility", "public") != "public"
                        or not is_chinese(it)):
                    continue                 # skip leads and non-Chinese voices
                blob = ((it.get("title", "") or "") + " "
                        + (it.get("description", "") or "") + " "
                        + " ".join(it.get("tags", []) or []))
                if _is_bad_voice(blob):
                    continue                 # skip sexy / breathy / ASMR voices
                g = _guess_voice_gender(blob)
                if g == "MALE" and vid not in males and len(males) < cap:
                    males.append(vid)
                elif g == "FEMALE" and vid not in females and len(females) < cap:
                    females.append(vid)
            if (len(males) >= cap and len(females) >= cap) or not data.get("has_more"):
                break

    # Prefer Fish's own Mandarin filter; if it is unsupported or yields too few,
    # sweep the most-used voices and keep the Chinese-looking ones.
    harvest({"language": "zh", "sort_by": "task_count"})
    if len(males) < 3 or len(females) < 3:
        harvest({"sort_by": "task_count"})
    return males, females


def search_fish_voices(fish_key, query="", language="", limit=25):
    """Search Fish's public voice library.

    Returns [{id, title, description, gender, languages}], newest-relevant
    first, with sexy/ASMR-marketed voices filtered out the same way the guest
    pool filters them. `gender` is Fish's metadata guessed from the title --
    Fish exposes no gender field -- so treat it as a hint, not a fact.

    The caller is responsible for only using voices they have the right to use;
    the library is public and Peiyin cannot verify what a clone is of.
    """
    if not fish_key:
        return []
    headers = {"Authorization": "Bearer " + fish_key}
    params = {"page_size": min(int(limit) * 2, 100), "page_number": 1,
              "sort_by": "task_count"}
    if query:
        params["title"] = query
    if language:
        params["language"] = language.split("-")[0].lower()
    try:
        with httpx.Client(timeout=60, trust_env=True) as c:
            r = c.get(FISH_MODEL_URL, headers=headers, params=params)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        items = (r.json() or {}).get("items", []) or []
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Fish voice search failed: {str(e)[:200]}")

    out = []
    for it in items:
        vid = it.get("_id")
        if not vid or it.get("visibility", "public") != "public":
            continue
        title = it.get("title", "") or ""
        desc = it.get("description", "") or ""
        tags = it.get("tags", []) or []
        blob = f"{title} {desc} {' '.join(tags)}"
        if _is_bad_voice(blob):
            continue
        out.append({"id": vid, "title": title[:60],
                    "description": desc[:80].replace("\n", " "),
                    "gender": _guess_voice_gender(blob) or "?",
                    "languages": ",".join(str(x) for x in (it.get("languages") or []))})
        if len(out) >= limit:
            break
    return out


PREVIEW_TEXT = {
    "zh-CN": "你好，很高兴认识你。这是这个声音的试听。",
}
DEFAULT_PREVIEW_TEXT = "Hello, nice to meet you. This is a preview of this voice."


def preview_line(language="zh-CN"):
    """A short sample sentence to audition a voice with."""
    return PREVIEW_TEXT.get(language, DEFAULT_PREVIEW_TEXT)


def preview_voice(fish_key, voice_id, text="", language="zh-CN", model=None):
    """Synthesise one short sample line. Returns the path to an mp3."""
    if not fish_key:
        raise RuntimeError("Paste your Fish Audio API key first.")
    if not (voice_id or "").strip():
        raise RuntimeError("No voice id to preview.")
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="peiyin-preview-")) / "preview.mp3"
    fish_tts(fish_key, text or preview_line(language), voice_id.strip(), out,
             model=model)
    return str(out)


def load_fish_auto_pool():
    """The cached public-voice pool. Empty unless the user opted in."""
    if not auto_public_voices_enabled():
        return [], []
    d = load_config().get("fish_auto_guest_pool", {}) or {}
    return d.get("MALE", []), d.get("FEMALE", [])


def ensure_fish_auto_pool(fish_key):
    """Fetch + cache the public guest-voice pool ONCE (like the transcript DB).

    OPT-IN: does nothing unless the user ticked "use public Fish voices for
    guests" in the Cast tab, because the public library can contain
    unauthorised clones of real people.

    Skipped entirely if the user's own pool already covers both genders. The
    pool is additionally screened by acoustic PITCH (see filter_pool_by_pitch)
    so shrill voices never reach a guest; older caches are upgraded in place.
    """
    if not auto_public_voices_enabled():
        return [], []
    m, f = load_fish_auto_pool()
    if m and f:
        cache = load_fish_voice_f0_cache()
        m2 = filter_pool_by_pitch(fish_key, m, cache)
        f2 = filter_pool_by_pitch(fish_key, f, cache)
        if [m2, f2] != [m, f]:              # a previously-cached shrill voice dropped
            save_config(fish_auto_guest_pool={"MALE": m2, "FEMALE": f2})
        return m2, f2
    pool = load_fish_guest_pool()
    if any(g == "MALE" for g, _ in pool) and any(g == "FEMALE" for g, _ in pool):
        return [], []                       # your pasted pool already covers it
    if not fish_key:
        return m, f
    males, females = fetch_fish_public_pool(fish_key)
    cache = load_fish_voice_f0_cache()
    males = filter_pool_by_pitch(fish_key, males, cache)
    females = filter_pool_by_pitch(fish_key, females, cache)
    if males or females:
        save_config(fish_auto_guest_pool={"MALE": males, "FEMALE": females})
    return males, females


MAX_GUEST_F0 = 300.0                          # Hz; median f0 above this -> too shrill
PITCH_PROBE_TEXT = "你好，很高兴认识你，今天天气真不错。"


def load_fish_voice_f0_cache():
    d = load_config().get("fish_voice_f0", {})
    return dict(d) if isinstance(d, dict) else {}


def probe_voice_f0(fish_key, voice_id, cache=None):
    """Median f0 (Hz) of a Fish voice, cached forever. None if it can't be
    measured (a probe/network failure NEVER rejects a voice -- fail open)."""
    if not voice_id:
        return None
    cache = load_fish_voice_f0_cache() if cache is None else cache
    if voice_id in cache:
        v = cache[voice_id]
        return float(v) if isinstance(v, (int, float)) and v > 0 else None
    if not fish_key:
        return None
    import tempfile as _tf
    d = Path(_tf.mkdtemp())
    mp3, wav = d / "probe.mp3", d / "probe.wav"
    f0 = None
    try:
        fish_tts(fish_key, PITCH_PROBE_TEXT, voice_id, mp3)
        to_std_wav(mp3, wav)
        f0 = estimate_median_f0(wav)
    except Exception as e:  # noqa: BLE001
        print(f"  ! pitch probe failed for {voice_id[:8]}: {str(e)[:80]}")
    finally:
        for p in (mp3, wav):
            try:
                p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
    if f0 and f0 > 0:
        cache[voice_id] = round(float(f0), 1)
        save_config(fish_voice_f0=cache)      # measured once, remembered forever
    return f0


def filter_pool_by_pitch(fish_key, ids, cache=None):
    """Drop voices whose measured median f0 exceeds MAX_GUEST_F0. Voices whose
    pitch can't be measured (no key / probe failure) are KEPT (fail open)."""
    cache = load_fish_voice_f0_cache() if cache is None else cache
    kept, dropped = [], []
    for vid in ids:
        f0 = probe_voice_f0(fish_key, vid, cache)
        if f0 is not None and f0 > MAX_GUEST_F0:
            dropped.append((vid, f0))
        else:
            kept.append(vid)
    if dropped:
        DIAG.append("Rejected " + str(len(dropped)) + " guest voice(s) as too "
                    "high-pitched (>" + f"{MAX_GUEST_F0:.0f}Hz): "
                    + ", ".join(f"{v[:8]}({f:.0f}Hz)" for v, f in dropped) + ".")
    return kept


def clean_emotion(s):
    """Normalise an emotion cue from the translator into a single short
    [bracket] tag, or \"\" if there is nothing usable. Kept OUT of the subtitle
    text -- it only ever prefixes the spoken text sent to Fish."""
    s = str(s or "").strip().replace("\n", " ").strip()
    if not s:
        return ""
    inner = s.strip("[]() ").strip()
    if not inner or len(inner) > 30:
        return ""
    return "[" + inner + "]"


# Casting is stored per profile, under the config dir -- never in the repo.
# Two shows can cast the same character name differently without colliding.

def _profile_key():
    from . import profiles
    return Path(profiles.active().path).stem or "default"


def profile_setting(key, default=None):
    """Read one per-profile setting from the user config file."""
    store = (load_config().get("by_profile") or {}).get(_profile_key()) or {}
    return store.get(key, default)


def save_profile_setting(**kw):
    """Write per-profile settings, leaving other profiles untouched."""
    cfg = load_config()
    store = dict(cfg.get("by_profile") or {})
    mine = dict(store.get(_profile_key()) or {})
    mine.update(kw)
    store[_profile_key()] = mine
    save_config(by_profile=store)


def load_voice_cast():
    """The user's chosen voice for each main character: {character: voice_id}.

    Empty by default. Peiyin ships NO voice ids: every voice is one the user
    picked, and they are responsible for only using voices they have the right
    to use. See the Cast tab.
    """
    return {k: v for k, v in (profile_setting("voice_cast") or {}).items() if v}


def save_voice_cast(cast):
    save_profile_setting(
        voice_cast={k: v.strip() for k, v in (cast or {}).items() if v and v.strip()})


def auto_public_voices_enabled():
    """Whether to draw guest voices from Fish's public library.

    OFF by default, and deliberately so: the public library can contain
    unauthorised clones of real people. The user opts in in the Cast tab and
    takes responsibility for what it returns.
    """
    return bool(profile_setting("auto_public_voices", False))


def set_auto_public_voices(on):
    save_profile_setting(auto_public_voices=bool(on))


def build_fish_voice_map():
    """Resolve a voice id for every speaker label.

    Main characters use the voice the user cast for them. OtherMale /
    OtherFemale (the generic guest buckets for unnamed one-off lines) take a
    pasted voice, else a public (auto) voice, preferring the right gender but
    accepting the other before ever touching a cast character's voice. They are
    NEVER one of the cast main characters."""
    out = load_voice_cast()
    reserved = set(out.values())            # a cast voice can never be a guest
    pool = load_fish_guest_pool()
    pasted_m = [v for g, v in pool if g == "MALE" and v not in reserved]
    pasted_f = [v for g, v in pool if g == "FEMALE" and v not in reserved]
    auto_m, auto_f = load_fish_auto_pool()
    auto_m = [v for v in auto_m if v not in reserved]
    auto_f = [v for v in auto_f if v not in reserved]
    any_m = pasted_m or auto_m
    any_f = pasted_f or auto_f
    out["OtherMale"] = (any_m or any_f or [""])[0]
    out["OtherFemale"] = (any_f or any_m or [""])[0]
    return out


def ensure_guest_voices(names, llm_key=""):
    """Recurring characters each keep their own voice for the whole series.
    Saved permanently, keyed by name."""
    reserved = set(speakers())
    names = sorted({n for n in names if n and n not in reserved and n != "Skip"})
    if not names:
        return {}
    key = "guest_voices"
    gmap = dict(profile_setting(key) or {})
    # Re-pick any guest currently stuck on a now-rejected (too shrill) voice, so a
    # bad earlier pick (e.g. Carol) is corrected from the pitch-filtered pool.
    f0cache = load_fish_voice_f0_cache()
    stale = [n for n, v in gmap.items()
             if isinstance(f0cache.get(v), (int, float))
             and f0cache.get(v) > MAX_GUEST_F0]
    for n in stale:
        gmap.pop(n, None)
    if stale:
        save_config(**{key: gmap})
        DIAG.append("Re-picking too-shrill guest voice(s): "
                    + ", ".join(sorted(stale)) + ".")
    unknown = [n for n in names if n not in gmap]
    if not unknown:
        return gmap

    # gender per new guest (asked once, then remembered)
    gend_key = "guest_genders"
    genders = dict(profile_setting(gend_key) or {})
    todo = [n for n in unknown if n not in genders]
    if todo and llm_key:
        try:
            model = pick_openai_model(llm_key) if llm_key.startswith("sk-") else None
            data = llm_json_model(
                llm_key,
                'You are given character names from a TV show. Reply with '
                'JSON only: {"genders":{"<name>":"MALE"|"FEMALE"}}. Guess '
                'sensibly from the name if unsure.',
                "Give the gender of each character: " + ", ".join(todo[:60]),
                model)
            for k, v in data.get("genders", {}).items():
                if v in ("MALE", "FEMALE"):
                    genders[k] = v
        except Exception as e:  # noqa: BLE001
            print("  ! guest gender lookup failed:", str(e)[:80])
    for n in unknown:
        genders.setdefault(n, "FEMALE" if n.lower().endswith(
            ("a", "ie", "y", "elle", "ine")) else "MALE")
    save_profile_setting(**{gend_key: genders})

    # ---- Fish Audio: every guest gets a voice, remembered forever. Priority:
    # voices you pasted, then popular PUBLIC Fish voices (fetched + cached once),
    # preferring the right gender but taking the other gender before ever using
    # a cast character's voice. Guests are NEVER one of the cast characters; if
    # the pool is completely empty the guest is left without a voice ("") and
    # the dub stops with a clear message rather than borrowing a cast voice.
    pool = load_fish_guest_pool()
    pasted_m = [v for g, v in pool if g == "MALE"]
    pasted_f = [v for g, v in pool if g == "FEMALE"]
    auto_m, auto_f = load_fish_auto_pool()
    other = build_fish_voice_map()          # resolved Other* ids (never cast)
    reserved = set(load_voice_cast().values())   # cast voices are untouchable
    used = set(gmap.values())

    f0c = load_fish_voice_f0_cache()

    def _ok_pitch(v):                        # keep unknown; reject only shrill
        f = f0c.get(v)
        return not (isinstance(f, (int, float)) and f > MAX_GUEST_F0)

    def pick_guest(name, want_f):
        h = int(hashlib.sha1(name.encode()).hexdigest(), 16)
        same = (pasted_f if want_f else pasted_m) + (auto_f if want_f else auto_m)
        opp = (pasted_m if want_f else pasted_f) + (auto_m if want_f else auto_f)
        cands = [v for v in (same or opp) if v not in reserved and _ok_pitch(v)]
        free = [v for v in cands if v not in used]
        if free:
            return free[h % len(free)]           # distinct, gender-matched
        if cands:
            return cands[h % len(cands)]         # reuse, still not cast
        return other["OtherFemale" if want_f else "OtherMale"] or ""

    for n in sorted(unknown):
        pick = pick_guest(n, genders.get(n) == "FEMALE")
        if pick:                         # never cache an empty assignment:
            gmap[n] = pick               # a transient empty pool must not
            used.add(pick)               # poison this guest forever
    save_profile_setting(**{key: gmap})
    return gmap


def fish_tts(fish_key, text, voice_id, path, emotion="", model=None):
    """Synthesise one line with Fish Audio and write MP3 bytes to `path`.
    `voice_id` is a Fish reference_id. `emotion` is an optional [bracket] cue
    that prefixes the spoken text (never shown in subtitles). S1 wants
    (parentheses) instead of [brackets], so convert if that model is chosen."""
    model = model or fish_model()
    emo = clean_emotion(emotion)
    if emo and model.startswith("s1"):
        emo = "(" + emo.strip("[]") + ")"
    say = (emo + " " + text).strip() if emo else text
    body = {"text": say, "reference_id": voice_id, "format": "mp3",
            "mp3_bitrate": 128, "sample_rate": 44100,
            "normalize": True, "latency": "normal"}
    headers = {"Authorization": "Bearer " + fish_key,
               "Content-Type": "application/json", "model": model}
    last = None
    for attempt in range(4):
        try:
            with httpx.Client(timeout=180, trust_env=True) as c:
                r = c.post(FISH_TTS_URL, headers=headers, json=body)
            if r.status_code == 200:
                if not r.content:
                    raise RuntimeError("Fish Audio returned no audio")
                Path(path).write_bytes(r.content)
                return
            if r.status_code in (429, 500, 502, 503):
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"Fish Audio HTTP {r.status_code}: {r.text[:300]}")
        except httpx.HTTPError as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Fish Audio request failed after retries: {last}")


def check_fish(fish_key):
    """Prove the Fish key works.

    If the user has already cast a voice, synthesise one tiny line with it --
    definitive, because it exercises the exact call the app makes. With nothing
    cast yet there is no voice to test, so fall back to an authenticated
    metadata call, which still proves the key is valid.
    """
    try:
        headers = {"Authorization": "Bearer " + fish_key,
                   "Content-Type": "application/json",
                   "model": fish_model()}
        probe = next(iter(load_voice_cast().values()), "")
        with httpx.Client(timeout=60, trust_env=True) as c:
            if probe:
                r = c.post(FISH_TTS_URL, headers=headers,
                           json={"text": "你好", "reference_id": probe,
                                 "format": "mp3"})
                if r.status_code == 200 and r.content:
                    return (f"FISH AUDIO: works ({len(r.content)} bytes for a "
                            f"test line).")
            else:
                r = c.get(FISH_MODEL_URL, headers=headers,
                          params={"page_size": 1, "page_number": 1})
                if r.status_code == 200:
                    return ("FISH AUDIO: key accepted. Cast a voice in the Cast "
                            "tab to test synthesis.")
        return f"FISH AUDIO: FAILED -> HTTP {r.status_code} {r.text[:180]}"
    except Exception as e:  # noqa: BLE001
        return f"FISH AUDIO: FAILED -> {str(e)[:180]}"


# --------------------------------------------------------------- provider API
# Everything the rest of Peiyin needs from a text-to-speech backend is declared
# here. Adding a second backend means implementing this interface and adding it
# to PROVIDERS -- no changes anywhere else.

@runtime_checkable
class TTSProvider(Protocol):
    """What Peiyin needs from a text-to-speech backend."""

    name: str

    def check_key(self, api_key: str) -> str:
        """Validate credentials. Returns a message for the user."""
        ...

    def synthesize(self, text, voice_id, path, *, api_key,
                   emotion="", model=None) -> None:
        """Render `text` in voice `voice_id` to `path`.

        `emotion` is an optional short delivery cue; a provider that does not
        support cues must ignore it rather than speak it. Raises on failure.
        """
        ...

    def search_voices(self, api_key, query="", language="", limit=25):
        """Search the provider's voice library.

        Returns [{id, title, description, gender, languages}]. A provider with
        no searchable library returns [].
        """
        ...

    def preview(self, api_key, voice_id, text="", language=""):
        """Synthesise a short sample line. Returns a path to an audio file."""
        ...


class FishAudioProvider:
    """Fish Audio -- the provider Peiyin ships with."""

    name = "fish"

    def check_key(self, api_key):
        return check_fish(api_key)

    def synthesize(self, text, voice_id, path, *, api_key,
                   emotion="", model=None):
        fish_tts(api_key, text, voice_id, path, emotion=emotion, model=model)

    def search_voices(self, api_key, query="", language="", limit=25):
        return search_fish_voices(api_key, query=query, language=language,
                                  limit=limit)

    def preview(self, api_key, voice_id, text="", language="zh-CN"):
        return preview_voice(api_key, voice_id, text=text, language=language)


#: Registered TTS providers, keyed by name.
PROVIDERS = {FishAudioProvider.name: FishAudioProvider()}
DEFAULT_PROVIDER = FishAudioProvider.name


def get_provider(name=None):
    """The TTS provider to use. Falls back to the default if `name` is unknown."""
    return PROVIDERS.get(name or DEFAULT_PROVIDER, PROVIDERS[DEFAULT_PROVIDER])
