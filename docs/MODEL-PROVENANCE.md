# Speaker model provenance

Verified 17 September 2026. Peiyin downloads this model at runtime; the repository
and wheel do not contain its weights.

- Artifact: `wespeaker_en_voxceleb_CAM++.onnx`, 29,292,684 bytes.
- Publisher: [k2-fsa/sherpa-onnx speaker recognition release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
- SHA-256: `c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef`.
- Original model: WeSpeaker CAM++ trained on VoxCeleb, by the WeSpeaker contributors.
- Weights license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated for VoxCeleb pretrained models in [WeSpeaker's model documentation](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md).
- Original ONNX named by the artifact's embedded metadata: `voxceleb_CAM++.onnx` at the WeSpeaker model host.

The sherpa-onnx [preparation script](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wespeaker/run.sh)
adds model metadata and renames that ONNX file to the exact artifact used here.
Its [release workflow](https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/export-wespeaker-to-onnx.yaml)
publishes the resulting ONNX files to the release tag above. We inspected the
actual downloaded artifact with ONNX Runtime: metadata identifies `framework`
as `wespeaker`, English at 16 kHz, output dimension 512, and the original
WeSpeaker URL. This connects the runtime download to the upstream model/license
statement rather than inferring a weights license from sherpa-onnx's code.

Attribution: WeSpeaker contributors, CAM++ VoxCeleb pretrained model; metadata
packaging by the sherpa-onnx contributors. Peiyin does not modify the downloaded
model and now verifies its checksum before loading it. No endorsement is implied.
Preserve the attribution, license link and change information if redistributing
these weights. The Apache-2.0 license of the runtime is separate.
