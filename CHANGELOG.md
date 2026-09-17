# 1.0.0b1 — private-use source release preparation

- Updated web dependencies; enforce loopback-only serving and private file boundaries.
- Fixed credential persistence, environment-key resolution and shared model selection.
- Added content/profile/model cache invalidation and optional editable script review.
- Aligned Python/Mandarin support, versioning, profile packaging and launchers.
- Added model checksum verification, attribution, personal-use disclaimer and guides.
- Added regression, synthetic rendering, HTTP boundary and wheel-content checks.
- Release scope is source-only; no public media demo or output upload required.

# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — unreleased

First public release. Peiyin began as a personal single-file tool hard-wired to
one show; this release is that tool rebuilt as a general one.

### Added

- **Show profiles.** A TOML file describes one show — characters, their
  aliases, and the user's own transcript source. Adding a show no longer means
  changing code. Ships with a documented `TEMPLATE.toml` and one worked example.
- **No-transcript mode.** A profile without transcripts still dubs, falling back
  to voice clustering, with a plain warning that accuracy is lower.
- **Transcript format parsers** for `name_colon`, `srt_with_speakers` and
  `html`, in a registry that makes a fourth one function.
- **Cast tab.** Per-character voice selection, a search over the TTS provider's
  voice library, and a Preview button that synthesises a sample line. Casting is
  stored per profile.
- **Settings tab.** Both API keys with their own "Test key" button, configurable
  model names, and the output folder. A first-run banner lists what is still
  missing before a dub can run.
- **Credentials from the environment.** `FISH_API_KEY`, `OPENAI_API_KEY` and
  `GEMINI_API_KEY`, or a `.env` file. An environment variable always wins over a
  stored key, so a key need never be written to disk.
- **A TTS provider interface**, so a second backend is an implementation rather
  than a rewrite.
- **A test suite** of 85 offline tests and CI across Linux, macOS and Windows on
  Python 3.10–3.12, including a hygiene job that fails the build on a committed
  voice id, API key, personal path or media file.

### Changed

- Split the 4,300-line single file into documented modules. `timing.py` is pure
  — plain data in, placements out — which is what makes the planner testable
  without audio.
- Fish Audio is the only TTS provider. The Google Chirp/Wavenet tiers and the
  Google Cloud Translation path were removed, leaving Fish plus OpenAI or Gemini
  as the only credentials.
- Config lives in `~/.peiyin` and output defaults to `~/Peiyin Output`. The
  config file is written owner-only (mode 600) because it can hold API keys.
- The Gradio file server no longer exposes the whole home directory.

### Removed

- **All default voice ids.** Earlier versions pinned a fixed set of voices.
  None ship: users choose their own, and are responsible for having the right
  to use them.
- **The bundled transcript source.** Transcript sources are configured by the
  user, per profile.
- **The automatic public-voice pool is now opt-in**, off by default, since
  Peiyin cannot verify the provenance of a voice in a public library.
- `edge-tts`, the only copyleft dependency, along with its voice tier.
- Dead code found during the port: two unreferenced casting prompts, an unused
  review prompt, and three unused constants.

### Known issues

- If the transcriber misjudges when a line starts, the dub starts at that wrong
  time. See Limitations in the README.
- Mandarin is the only tested target language.
- The theme-song muting that earlier versions applied to one specific show is
  now profile-driven and ships empty; fill in `theme_words` or
  `skip_intro_seconds` in your own profile to use it.

---

## Pre-release history

Peiyin was a personal tool before this release. Condensed, in order:

- **v5** — A line plays into empty time. Instead of being capped shortly after
  its own words and refusing to cross a silence, a line now runs at natural
  speed for as long as it needs, stopping only for the next speaker's audio.
  Drift became the preferred remedy and speed-up the last resort. This is the
  timing behaviour the current planner still implements.
- **v4** — Replaced energy-based edge detection, which mistook a laugh track and
  music for speech, with the neural Silero VAD bundled in faster-whisper. Line
  ends extend to where speech truly stops.
- **v3** — Energy-based "snapping" of line edges. Reverted in v4.
- **v2** — Word-level onsets and offsets, so a line's window comes from its
  first and last word rather than a loose segment box, and a short line can no
  longer be stretched across a scene. Added the silence map and the
  backward-deadline scheduler that lets a line borrow a later line's slack.
- **Earlier** — Moved TTS to Fish Audio for expressive Mandarin with delivery
  cues; added the script-first spine so every transcript line is dubbed exactly
  once; added voice-embedding clustering as the speaker fallback; added the
  editable script review table and the batch queue.
