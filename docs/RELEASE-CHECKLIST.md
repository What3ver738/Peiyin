# Source-only release plan

Publish the code and documentation as a personal-use beta. **No demo, example
video or generated dub needs to be uploaded.** Private testing and public
publication are separate activities.

## Implemented

- [x] Personal-use disclaimer, privacy explanation and local-use security notes.
- [x] Current web stack, loopback binding, sharing/analytics disabled, restricted downloads.
- [x] Atomic private credential saves; environment keys work without persistence.
- [x] Content/profile/transcript/model cache boundaries and TTS model invalidation.
- [x] Optional script review with persisted edits and stale-review rejection.
- [x] Consistent casting/translation model selection without account probing.
- [x] Explicit Mandarin-only validation, supported-Python launchers and shared beta version.
- [x] Packaged profile templates, notices and model attribution; checksum verification.
- [x] Regression tests, synthetic rendering check and local HTTP boundary check.
- [x] CI for tests, vulnerability audit, HTTP smoke, hygiene and package contents.

## Before publishing

- [ ] Confirm the code is yours to publish, including any employment/school rules
      and third-party code attribution. A source scan cannot establish ownership.
- [ ] Review the diff and historical commit author email; keep private information out.
- [ ] Run one short **private** dub with media and voices you may lawfully use when
      convenient. Keep the output local. Until then, retain the documented statement
      that current paid-API compatibility and actual dubbing quality are unverified.
- [ ] Choose the GitHub account and create an empty private repository named `peiyin`.

## Upload through GitHub in your browser

1. Click **Create repository**. Name it `peiyin` and choose **Private** for the
   initial upload. Leave the options to add a README, `.gitignore` and license off;
   the prepared files already include these. Click **Create repository**.
2. On the empty repository page, click **uploading an existing file**.
3. In Finder, open the prepared `peiyin-upload` folder beside the original
   project. Press **Command–Shift–.** to show hidden files. Select the folder's
   **contents**, including `.github`, `.gitignore` and `.env.example`, and drag
   them into the upload area. Do not upload the enclosing folder or a ZIP.
4. Check that `README.md`, `LICENSE`, `DISCLAIMER.md`, `pyproject.toml` and the
   `peiyin`, `docs`, `scripts` and `tests` directories appear at the repository
   root. The prepared folder excludes Git history, real keys, caches and media.
5. Enter “Initial source release” as the commit message and commit the files.
6. Check that the README renders and the **Actions** checks pass. Investigate
   failures before publishing. The local checks do not substitute for CI on
   the other supported operating systems.
7. When ready, open **Settings → General → Danger Zone → Change repository
   visibility**, choose **Public**, and follow GitHub's confirmation prompts.
8. Pin the repository on your GitHub profile and link it from your CV.

Suggested description: “Local Mandarin-dubbing tool for private language learning,
with speech recognition, speaker attribution, translation and timed synthesis.”

A public demo, PyPI upload, executable bundle or release tag is unnecessary.
GitHub's browser upload supports up to 100 files at once and 25 MiB per file;
this source-only folder fits those limits. See GitHub's
[upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

## Verification

See [PUBLISHING-REVIEW.md](PUBLISHING-REVIEW.md) for dated results and limitations.
Nothing has been pushed or published by these local changes. No paid API calls
or private media have been used for automated verification.

## CV wording

“Built Peiyin, a Python Mandarin-dubbing tool integrating speech recognition,
speaker attribution, LLM translation and timed speech synthesis, with a Gradio
interface, editable script review and automated tests.”

Describe your own contribution accurately in an interview, including how you
used Claude. Do not claim production readiness, benchmark accuracy or authorship
of the underlying models. A readable repository is sufficient; a public media
demo is not part of this release plan.
