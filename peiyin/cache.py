"""Content and configuration identities for independently reusable stages."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from . import profiles
from .config import GEMINI_MODEL, OPENAI_MODEL, llm_model
from .util import file_key

# Bump when casting/translation algorithms or prompts change.
SCRIPT_REVISION = 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def profile_fingerprint(profile=None, use_scripts=True):
    prof = profile or profiles.active()
    data = asdict(prof)
    data.pop("path", None)
    if not use_scripts:
        data.pop("transcripts", None)
    elif prof.transcripts.local_folder:
        folder = Path(prof.transcripts.local_folder).expanduser()
        if not folder.is_dir():
            raise RuntimeError("The configured transcript folder does not exist.")
        data["local_files"] = [(f.name, file_key(f)) for f in sorted(folder.iterdir())
                               if f.is_file() and not f.name.startswith(".")]
    return fingerprint(data)


def script_workdir(workdir, model_size, key, use_scripts=True):
    """Audio/ASR stay reusable; casting, translation and edits share this scope.

    Remote transcript changes are explicit through transcripts.revision. No
    credential or credential hash is written into cache keys or manifests.
    """
    provider = "openai" if key.startswith("sk-") else "gemini"
    model = llm_model() or (OPENAI_MODEL if provider == "openai" else GEMINI_MODEL)
    identity = dict(revision=SCRIPT_REVISION,
                    profile=profile_fingerprint(use_scripts=use_scripts),
                    asr=model_size, provider=provider, model=model,
                    transcripts=bool(use_scripts))
    return Path(workdir) / "scripts" / fingerprint(identity)
