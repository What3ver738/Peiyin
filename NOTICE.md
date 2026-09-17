# Third-party notices

Peiyin is MIT licensed. It builds on the following third-party software and
services, each under its own terms.

## Libraries

| Component | License | Link |
|---|---|---|
| faster-whisper | MIT | https://github.com/SYSTRAN/faster-whisper |
| Silero VAD (bundled with faster-whisper) | MIT | https://github.com/snakers4/silero-vad |
| CTranslate2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| sherpa-onnx | Apache-2.0 | https://github.com/k2-fsa/sherpa-onnx |
| ONNX Runtime | MIT | https://github.com/microsoft/onnxruntime |
| Gradio | Apache-2.0 | https://github.com/gradio-app/gradio |
| FastAPI | MIT | https://github.com/fastapi/fastapi |
| Starlette | BSD-3-Clause | https://github.com/encode/starlette |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| soundfile | BSD-3-Clause | https://github.com/bastibe/python-soundfile |
| imageio-ffmpeg | BSD-2-Clause | https://github.com/imageio/imageio-ffmpeg |
| FFmpeg (binary, via imageio-ffmpeg) | Build-dependent: LGPL or GPL; inspect the actual binary | https://ffmpeg.org/legal.html |

## Models

| Model | License | Link |
|---|---|---|
| WeSpeaker CAM++ speaker embedding (`wespeaker_en_voxceleb_CAM++`), downloaded at runtime | CC BY 4.0; see model provenance and attribution | https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md |
| Whisper models, downloaded at runtime by faster-whisper | MIT | https://github.com/openai/whisper |

## Services

These are third-party APIs you supply your own credentials for. Peiyin sends
them your transcript text and receives audio or translations in return. Their
terms and pricing are your responsibility.

| Service | Used for | Terms |
|---|---|---|
| Fish Audio | text-to-speech | https://fish.audio |
| OpenAI | speaker casting, translation | https://openai.com/policies |
| Google Gemini | speaker casting, translation | https://ai.google.dev/terms |

No show, transcript, subtitle or voice content is distributed with this project.

Model attribution, exact download identity and verified preparation chain are
recorded in [MODEL-PROVENANCE.md](docs/MODEL-PROVENANCE.md).

This table is a starting inventory, not a complete distribution compliance audit.
Python package metadata does not establish the license of embedded FFmpeg or
libsndfile binaries. A source-only release should not attach environments,
model weights or executable bundles. Binary releases need their own notices,
license texts and any applicable source-code obligations.

Verified on 17 September 2026: the macOS arm64 imageio-ffmpeg 0.6.0 wheel
installed for this review contains FFmpeg 7.1 built with `--enable-gpl`; its
`-L` output states GPL v2 or later. Other platform builds must be checked
individually. This does not by itself change the license of Peiyin source code.
