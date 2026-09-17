"""Transcript loading and episode identification.

Parsers for the supported transcript formats, the transcript cache, and the
trigram scoring that works out which episode a video is.

Transcript *sources* are configured per profile by the user
no source ships
with this project.
"""

import concurrent.futures
import html as _htmlmod
import json
import re
import time
from pathlib import Path

import httpx

from . import profiles
from .cache import fingerprint, profile_fingerprint
from .config import SCRIPT_DIR

TAG_RE = re.compile(r"<[^>]+>")
PAREN_RE = re.compile(r"\([^)]*\)")
BRACKET_RE = re.compile(r"\[[^\]]*\]")


def _clean(s):
    s = TAG_RE.sub(" ", s)
    s = _htmlmod.unescape(s)
    s = BRACKET_RE.sub(" ", s)
    s = PAREN_RE.sub(" ", s)
    s = s.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


def _clean_script_text(text):
    import html as _html
    text = _html.unescape(text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


NOT_NAMES = {
    "and", "the", "you", "yes", "no", "oh", "ok", "okay", "but", "so", "if",
    "am", "is", "are", "was", "were", "that", "this", "what", "why", "how",
    "who", "when", "where", "not", "huge", "very", "really", "just", "now",
    "here", "there", "all right", "i", "me", "my", "we", "he", "she", "it",
    "they", "her", "him", "his", "hers", "do", "did", "does", "don't",
    "didn't", "can", "can't", "will", "won't", "never", "always", "more",
    "less", "yeah", "hey", "hi", "hello", "wow", "god", "please", "thanks",
    "thank you", "sorry", "wait", "stop", "look", "listen", "one", "two",
    "end", "note", "time lapse", "cut to", "flashback",
}


def _valid_speaker(raw):
    t = re.sub(r"\(.*?\)", "", raw).strip().strip(".").strip().rstrip(":").strip()
    if not t or len(t) > 40 or any(ch.isdigit() for ch in t):
        return None
    if re.search(r"scene|credits|note|time lapse|cut to|commercial|written by"
                 r"|transcribed|opening|closing", t, re.I):
        return None
    low = t.lower()
    if low in NOT_NAMES:
        return None
    if len(t) < 2 or len(t.split()) > 4:
        return None
    if not re.search(r"[A-Za-z]{2}", t):
        return None
    return t


def _parse_bold(page):
    # A speaker label is bold text followed by a colon (inside or just after
    # the tag). Bold is also used for emphasis inside dialogue, which must
    # NOT be mistaken for a character.
    marker = re.compile(
        r"<(?:b|strong)\b[^>]*>\s*(?:<[^>]+>\s*)*([^<:]{1,45}?)\s*(:?)\s*"
        r"(?:</[^>]+>\s*)*(:?)\s*</(?:b|strong)>(\s*:)?", re.I)
    out, matches = [], [m for m in marker.finditer(page)
                        if m.group(2) or m.group(3) or m.group(4)]
    for mi, m in enumerate(matches):
        nxt = matches[mi + 1].start() if mi + 1 < len(matches) else len(page)
        chunk = page[m.end():nxt]
        chunk = re.split(r"(?i)\[\s*scene|\bcommercial break\b"
                         r"|\bopening credits\b|\bclosing credits\b", chunk)[0]
        text = _clean_script_text(re.sub(r"<[^>]+>", " ", chunk))
        sp = _valid_speaker(m.group(1))
        if sp and len(text) >= 2:
            out.append({"speaker": sp, "text": text})
    return out


def _parse_plain(page):
    import html as _html
    txt = _html.unescape(re.sub(r"<[^>]+>", "\n", page))
    out, cur, buf = [], None, []

    def flush():
        if cur and buf:
            t = _clean_script_text(" ".join(buf))
            if t:
                out.append({"speaker": cur, "text": t})

    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Z][A-Z'.\- ]{1,30}?):\s*(.*)$", line)
        if m:
            flush()
            cur = _valid_speaker(m.group(1))
            buf = [m.group(2)] if cur else []
        elif cur:
            if re.match(r"^[\[(]", line):
                flush()
                cur, buf = None, []
            else:
                buf.append(line)
    flush()
    return out


def fetch(url, timeout=45):
    """Plain GET with retries; used for the transcript library."""
    last = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout, trust_env=True,
                              follow_redirects=True) as c:
                r = c.get(url)
            if r.status_code == 200:
                return r.text
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url.split('/')[-1]}: {last}")


def norm_words(text):
    """Words used for matching: lowercase, letters only, no filler."""
    return [w for w in re.findall(r"[a-z']+", (text or "").lower())
            if len(w) > 1]


# Character names and their aliases come from the active profile; see
# profiles/TEMPLATE.toml. Everything else in this file is language-generic.
GROUP_WORDS = {"all", "everyone", "everybody", "guys", "girls", "both",
               "together", "gang", "group", "the girls", "the guys"}


DIFFERENT_PERSON_RE = re.compile(
    r"('s\b|\u2019s\b|\b(?:fake|young|old|baby|mini|future|evil|imposter|"
    r"double|mrs|mr|ms|dr|miss|grandmother|grandma|granny|grandpa|grandfather|"
    r"mom|mother|dad|father|sister|brother|aunt|uncle|cousin|niece|nephew|"
    r"sr|jr|senior|junior)\b)")


def canonical_speaker(raw):
    """Script speaker label -> a main character, GROUP, or GUEST.

    Only an EXACT name (or known alias from the profile) is a main character.
    A possessive or a relative/look-alike marker ("Alice's Grandmother", "Fake
    Alice", "Alice's Date") is a different person and must never be given a
    main character's voice.
    """
    prof = profiles.active()
    MAIN_ALIASES = prof.aliases
    MAIN_FULLNAMES = prof.full_names
    n = re.sub(r"\(.*?\)", "", (raw or "").lower()).strip().rstrip(":").strip()
    n = n.strip('."\'')
    if not n:
        return "GUEST"
    # a possessive or relative/look-alike marker => a DIFFERENT person, never lead
    if DIFFERENT_PERSON_RE.search(n):
        return "GUEST"
    if n in MAIN_ALIASES:
        return MAIN_ALIASES[n]
    if n in MAIN_FULLNAMES:
        return MAIN_FULLNAMES[n]
    if n in GROUP_WORDS:
        return "GROUP"
    # compound: "Alice and Bob", "Alice, Bob" -> first named member
    parts = [x.strip() for x in re.split(r"\s*(?:,|\band\b|&|\+|/)\s*", n) if x.strip()]
    if len(parts) > 1:
        members = [MAIN_ALIASES[x] for x in parts if x in MAIN_ALIASES]
        if members:
            return members[0] if len(members) == 1 else "GROUP"
        if all(x in GROUP_WORDS for x in parts):
            return "GROUP"
        return "GUEST"
    # single label whose FIRST token is a main character's name plus a surname
    # or initial ("Alice Smith", "Alice S.") -- different-person markers were
    # already ruled out above, so this is safe.
    toks = n.split()
    if toks and toks[0] in MAIN_ALIASES and len(toks) <= 3:
        return MAIN_ALIASES[toks[0]]
    return "GUEST"


def parse_html(page):
    """A web page. Two layouts are common (bold-tag and plain CAPS:); use
    whichever yields more dialogue."""
    page = re.sub(r"(?is)<script.*?</script>", " ", page)
    a = _parse_bold(page)
    b = _parse_plain(page)
    return a if len(a) >= len(b) else b


def parse_name_colon(text):
    """One utterance per line, "NAME: dialogue".

    A line without a speaker prefix continues the previous speaker, which is how
    wrapped paragraphs usually appear.
    """
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^([^:]{1,45}?)\s*:\s*(.*)$", line)
        if m and _valid_speaker(m.group(1)):
            body = _clean_script_text(m.group(2))
            if len(body) >= 2:
                out.append({"speaker": _valid_speaker(m.group(1)), "text": body})
        elif out:
            body = _clean_script_text(line)
            if body:
                out[-1]["text"] = (out[-1]["text"] + " " + body).strip()
    return out


SRT_INDEX_RE = re.compile(r"^\d+$")
SRT_TIME_RE = re.compile(r"-->")


def parse_srt_with_speakers(text):
    """An .srt subtitle file whose cue text starts with "NAME: ".

    Index and timing lines are dropped
    consecutive cues from one speaker are
    joined, because a single utterance is usually split across several cues.
    """
    out = []
    for block in re.split(r"\n\s*\n", (text or "").replace("\r\n", "\n")):
        body = []
        for line in block.splitlines():
            line = line.strip()
            if not line or SRT_INDEX_RE.match(line) or SRT_TIME_RE.search(line):
                continue
            body.append(line)
        if not body:
            continue
        joined = " ".join(body)
        m = re.match(r"^[-\s]*([^:]{1,45}?)\s*:\s*(.*)$", joined)
        speaker = _valid_speaker(m.group(1)) if m else None
        content = _clean_script_text(m.group(2) if m else joined)
        if len(content) < 2:
            continue
        if speaker:
            out.append({"speaker": speaker, "text": content})
        elif out:
            out[-1]["text"] = (out[-1]["text"] + " " + content).strip()
    return out


#: Transcript format name -> parser. Adding a format is one function plus one
#: entry here and in profiles.KNOWN_FORMATS.
PARSERS = {
    "name_colon": parse_name_colon,
    "srt_with_speakers": parse_srt_with_speakers,
    "html": parse_html,
}
assert set(PARSERS) == set(profiles.KNOWN_FORMATS), (
    "episodes.PARSERS and profiles.KNOWN_FORMATS have drifted apart")


def episode_codes(transcripts=None):
    """Which episode ids a url_base source is expected to have.

    Only meaningful for `url_base`: a local folder or an explicit url_list
    already names its episodes.
    """
    t = transcripts if transcripts is not None else profiles.active().transcripts
    codes = [f"{se:02d}{ep:02d}"
             for se, n in enumerate(t.episodes_per_season, start=1)
             for ep in range(1, int(n) + 1)]
    covered = set(t.covered_by_multi)
    return [c for c in codes if c not in covered] + list(t.multi_part_pages)


PARSER_VERSION = 4

#: A transcript file yielding fewer speaker lines than this is treated as
#: mis-parsed rather than as a very short episode -- almost always the wrong
#: `format` setting for the files.
MIN_TRANSCRIPT_LINES = 5


def _cache_dir(profile):
    """Transcripts are cached per profile, so two shows never mix."""
    return SCRIPT_DIR / fingerprint([PARSER_VERSION, profile_fingerprint(profile)])


def _load_local(folder, parse, progress=None):
    """Read every transcript file in a folder. The file name is the episode id."""
    d = Path(folder).expanduser()
    if not d.is_dir():
        raise RuntimeError(
            f"The transcript folder set for this profile does not exist: {d}")
    files = sorted(f for f in d.iterdir()
                   if f.is_file() and not f.name.startswith("."))
    out = {}
    for n, f in enumerate(files, start=1):
        try:
            lines = parse(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            lines = []
        if len(lines) >= MIN_TRANSCRIPT_LINES:
            out[f.stem] = lines
        if progress:
            progress(n / max(len(files), 1),
                     desc=f"Reading transcripts... {n}/{len(files)}")
    if not out:
        raise RuntimeError(
            f"No readable transcripts in {d}. Check the profile's `format` "
            f"setting -- it is currently \"{profiles.active().transcripts.fmt}\".")
    return out


def _load_remote(urls, parse, progress=None):
    """Fetch transcripts over the network. Returns {episode id: lines}."""
    out = {}
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        def grab(item):
            code, url = item
            try:
                return code, parse(fetch(url))
            except Exception:  # noqa: BLE001
                return code, None
        for code, lines in pool.map(grab, urls):
            done += 1
            if lines and len(lines) >= MIN_TRANSCRIPT_LINES:
                out[code] = lines
            if progress:
                progress(done / max(len(urls), 1),
                         desc=f"Building the transcript database "
                              f"(one time)... {done}/{len(urls)}")
    return out


def _remote_urls(t):
    """[(episode id, url)] for whichever remote source the profile configured.

    `url_list` is taken literally, the episode id coming from the file name.
    `url_base` is a prefix that an episode id and `url_suffix` are appended to,
    so ("https://example.com/ep/", "0101", ".html") makes
    "https://example.com/ep/0101.html". The first base is used; a profile can
    list more as mirrors.
    """
    if t.url_list:
        return [(Path(u).stem or f"{i:04d}", u)
                for i, u in enumerate(t.url_list, 1)]
    if not t.url_base:
        return []
    base = t.url_base[0]
    return [(code, f"{base}{code}{t.url_suffix}") for code in episode_codes(t)]


def ensure_scripts(progress=None):
    """Build this profile's transcript database ONCE, then reuse it.

    Local content, profile configuration and parser changes select a new cache.
    For updated remote content, change transcripts.revision in the profile.
    """
    prof = profiles.active()
    t = prof.transcripts
    if not t.configured:
        raise RuntimeError(
            "This profile has no transcript source, so episodes cannot be "
            "identified. Set `local_folder`, `url_list` or `url_base` in the "
            "profile, or run without transcripts (lower speaker accuracy).")
    parse = PARSERS[t.fmt]
    cache = _cache_dir(prof)
    cache.mkdir(parents=True, exist_ok=True)

    vfile = cache / "_parser_version"
    try:
        cached_v = int(vfile.read_text(encoding="utf-8").strip())
    except Exception:  # noqa: BLE001
        cached_v = 0
    if cached_v != PARSER_VERSION:
        for f in cache.glob("*.json"):
            f.unlink(missing_ok=True)
        vfile.write_text(str(PARSER_VERSION), encoding="utf-8")

    have = sorted(f.stem for f in cache.glob("*.json")
                  if not f.stem.startswith("_"))
    if have:
        return have

    if t.local_folder:
        found = _load_local(t.local_folder, parse, progress)
    else:
        found = _load_remote(_remote_urls(t), parse, progress)

    for code, lines in found.items():
        (cache / f"{code}.json").write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")

    have = sorted(found)
    if not have:
        raise RuntimeError(
            "No transcripts could be read for this profile. Check the "
            "transcript source and `format` setting, and your network "
            "connection if the source is a URL.")
    return have


def _trigrams(lines, key):
    out = set()
    for ln in lines:
        ws = norm_words(ln[key])
        out.update(tuple(ws[k:k + 3]) for k in range(len(ws) - 2))
    return out


def identify_episode(segments):
    """Which episode is this? Scored by how much of what Whisper heard the
    script explains. Returns (episode_id, script_lines, score, runner_up)."""
    heard = _trigrams(segments, "en")
    if not heard:
        raise RuntimeError("nothing was transcribed")
    scores = []
    # Transcripts are cached per profile, so only this show's are considered.
    for f in sorted(_cache_dir(profiles.active()).glob("*.json")):
        if f.stem.startswith("_"):
            continue
        try:
            lines = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not lines:
            continue
        sc = len(heard & _trigrams(lines, "text")) / len(heard)
        scores.append((sc, f.stem, lines))
    if not scores:
        raise RuntimeError("no transcripts available")
    scores.sort(key=lambda x: -x[0])
    best, second = scores[0], (scores[1] if len(scores) > 1 else (0, "", []))
    return best[1], best[2], best[0], second[0]
