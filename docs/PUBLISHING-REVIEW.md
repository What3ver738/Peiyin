# Release fixes and verification

Updated 17 September 2026. The earlier review covered commit `0096f65`; this
report describes the implemented changes on top of it. Release scope is now a
**source-only personal-use beta, without a public media demo**.

## Resolved findings

| Earlier issue | Implemented change |
|---|---|
| Old Gradio/FastAPI/Starlette stack | Updated to Gradio 6.27.0, FastAPI 0.141.1 and Starlette 1.6.0, with current Hugging Face Hub and a patched Pillow floor. Removed the old schema monkeypatches. |
| Broad file serving / accidental exposure | Explicit loopback binding, sharing disabled, no Gradio analytics/monitoring/MCP/run history. Only dedicated result/preview copies are allowed; private app storage is blocked. |
| Cached scripts survive profile/model/input changes | Full SHA-256 input identity; profile and local transcript content, ASR model, LLM provider/model and transcript mode select separate script/edit caches. Remote sources have an explicit revision field. |
| Synthesis model ignored by audio cache | Fish model is included in the synthesis stamp. Relevant voice/text/emotion changes also invalidate audio. |
| Missing review screen | Added Prepare script → edit speaker/Mandarin → Render reviewed script. Edits are scoped to their video/profile/model; stale review state is rejected. |
| Different model used for translation | One selected/default model now controls casting and translation. Removed paid account probing and automatic stronger-model selection. |
| Non-atomic credential writes | Central atomic replacement using a private temporary file, with save/read failures surfaced. Removed direct config writes from UI callbacks. |
| Environment keys / saved keys in UI | Environment keys work in key checks, casting previews and dubbing without being saved. Stored keys are not prefilled in browser config; only Settings explicitly saves entered keys. |
| Unsupported language / Python claims | Non-Mandarin profiles are rejected. Metadata and launchers consistently support Python 3.10–3.12. UI and package share version 1.0.0b1. |
| Missing profile data in wheel | Moved templates into package data. Wheel and source distribution contain required profiles and notices. |
| Incorrect model / FFmpeg license assertions | Added the verified model provenance and CC BY 4.0 attribution; verify the exact runtime download by checksum. FFmpeg notices reflect the actual GPL-enabled Mac binary and platform dependence. |
| Misleading local-only / no-download documentation | README and guide accurately describe cloud text processing, model downloads, optional transcript fetching and local storage. |
| Demo requirement conflicts with private use | Removed it. Added DISCLAIMER.md; the repository can demonstrate engineering through code, tests and architecture alone. |

The local UI serializes its operations so a profile/settings change cannot run
in the middle of a render. Exports contain a content identifier to distinguish
same-named videos. MP4 metadata identifies synthetic dubbing; it does not claim
to satisfy every disclosure/marking obligation.

## Verification results

| Check | Observed result |
|---|---|
| Unit/regression suite | 99 passed on macOS arm64, Python 3.12 |
| Synthetic rendering integration | Generated video/tones exported through actual FFmpeg/mixing; repeated render reused synthesis and changing Fish model regenerated it |
| UI review callbacks | Edits reached rendering and persisted; changed input was rejected |
| Live local HTTP check | UI/config/upload/download succeeded; saved key absent from config response; direct config/work/unrelated files denied |
| Loopback enforcement | Overrode a test `GRADIO_SERVER_NAME=0.0.0.0` with 127.0.0.1; no public share URL |
| Dependency consistency | `pip check`: no broken requirements |
| Vulnerability audit | `pip-audit`: no known vulnerabilities in the updated local environment at the time checked |
| Package build | Wheel and sdist built; wheel installed outside the repository and created an editable profile; profiles/notices verified; no media/model weights in wheel |
| Paid provider workflow | Not run; no real keys, private media or paid API calls used in automated checks |
| Other OS/Python combinations | CI configured for Windows/Linux/macOS and Python 3.10–3.12; not claimed passed until GitHub runs it |

No automated result proves dubbing quality, provider availability, voice consent,
content rights or universal security. A private real-API check can be done later;
until then the README explicitly states that limitation. No public demo is needed.

## Rights and publication

The [personal-use disclaimer](../DISCLAIMER.md) states intended use, required
rights/consent, cloud processing, limitations and the existing MIT warranty terms.
It does not turn private/educational use into a blanket legal exemption, and it
does not add a noncommercial restriction to the MIT source license.

Copyright owners generally control translation/adaptation, subject to applicable
exceptions. Avoid including copyrighted media or third-party transcripts in the
repository. [WIPO copyright FAQ](https://www.wipo.int/en/web/copyright/faq-copyright).
Fish library availability and an API subscription are not proof of rights to a
particular voice. [Fish terms](https://fish.audio/terms/).

Model identity, attribution and the upstream preparation chain are documented in
[MODEL-PROVENANCE.md](MODEL-PROVENANCE.md). FFmpeg licensing depends on build
options; a future executable bundle needs a separate distribution audit.
[FFmpeg legal guidance](https://ffmpeg.org/legal.html).

If EU rules apply, assess relevant AI Act roles and obligations separately.
The Commission distinguishes provider marking from deployer disclosure; a
personal-use statement, source license or generic MP4 metadata cannot certify
compliance. No jurisdiction-specific legal opinion is claimed here.
[Commission Article 50 guidance](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act).

The author still needs to confirm ownership/provenance of the source and any
school/employer obligations. Git history and commit metadata should be reviewed
before the public push. The initial pattern scan covered 12 reachable commits
and 130 unique blobs, finding no matching API-key patterns or media/model file
extensions. This is a limited hygiene check, not proof of legal clearance.

## Technical references

- [Gradio 6 migration guide](https://gradio.app/guides/gradio-6-migration-guide): migrated app-level styling to launch and updated dataframe configuration.
- [OpenAI Chat API reference](https://developers.openai.com/api/reference/resources/chat): retained the existing JSON-mode chat interface, using the same model selection on both paths. Real account compatibility remains untested.

See [RELEASE-CHECKLIST.md](RELEASE-CHECKLIST.md) for the remaining source upload steps.
