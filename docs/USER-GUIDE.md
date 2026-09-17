# Your first dub with Peiyin

Start with a 30–60 second English video you recorded yourself and have permission
to process. The current translation output is Mandarin. Use an original script,
no third-party background music, and authorized voices.

## 1. Install

Install Python **3.12**, then download the repository using GitHub's **Code →
Download ZIP** and extract it, or clone the published repository. Open a terminal
in the folder containing `requirements.txt`.

macOS / Linux:

```bash
python3.12 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m peiyin
```

Windows PowerShell:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m peiyin
```

The browser opens the local interface. Keep the terminal running. Close the app
with Ctrl+C in that terminal. The first run downloads dependencies and speech
models, so it takes longer. The OS-specific launchers in `scripts/` are optional.
The app binds to localhost with sharing disabled. Keep it local; do not use a
tunnel, public server or shared-machine deployment.

## 2. Add your keys

In **Settings**, enter your Fish Audio API key and either an OpenAI or Gemini API
key. Press **Test key**, then **Save settings**. API access is separate from
consumer chat subscriptions. Check your providers' current billing and set any
available spending limits before running a queue. Voice previews and key
tests can also make paid requests.

Alternatively, set `FISH_API_KEY` and `OPENAI_API_KEY` or `GEMINI_API_KEY` in your
environment. `.env.example` lists the supported settings. Environment variables
take precedence. Keys saved through the UI are plaintext in
`~/.peiyin/config.json`; file permissions are not encryption.

## 3. Create a profile

In **Dub**, create a profile, then edit its TOML file under `~/.peiyin/profiles/`.
Use `peiyin/data/profiles/TEMPLATE.toml` as the reference. Keep `target_language = "zh-CN"`.
Define your character names and aliases, and save the file. Restart Peiyin after
editing and select your profile.

For better speaker attribution, create your own transcript with at least five
speaker-labelled lines. Put one episode per text file, such as `0101.txt`, in a
folder outside the repository. A minimal original example:

```text
Alice: I packed a notebook for our walk.
Bob: Did you remember the map?
Alice: Yes, it is in the blue bag.
Bob: Then let us take the path by the river.
Alice: We can stop at the bridge and draw the view.
Bob: I will bring an extra pencil.
```

In the profile, set `local_folder` to that folder and `format = "name_colon"`.
On Windows, use forward slashes in TOML paths, such as `C:/Users/you/transcripts`.
Leave URL fields empty. A profile without transcripts is supported, but speaker
attribution is less reliable and may require guest voices.

## 4. Cast and preview

Open **Cast**. Assign an authorized Fish voice reference ID to each main
character and press **Save cast**. Preview a short sample. Configure authorized
guest voices for any additional speakers. Leave automatic public-library guest
selection off. Library availability alone does not establish consent.

## 5. Run one clip

In **Dub**, add your video and press **Dub it (one episode)**. The app processes
it automatically. For an optional pause, open **Review script → Prepare script**,
edit the Speaker or Mandarin columns, then press **Render reviewed script**. Use
`Skip` to mute a line. Changes to the video, profile or processing settings require
preparing again. Start with one file, not a season.

Exports go to `~/Peiyin Output` unless changed in Settings:

- `your-file-CONTENT_ID.chinese.fish.mp4`: dubbed video;
- `your-file-CONTENT_ID.chinese.srt`: subtitle file.

Play the entire short output. Check translation, speaker assignment, timing,
missing audio and subtitle rendering. The original soundtrack is ducked rather
than separated into clean dialogue/music tracks, so original speech may remain
audible. If a line cannot be voiced, it may survive only in subtitles.

## 6. Re-run and troubleshoot

- **Re-runs:** video content, profile/transcript changes and model changes now
  select fresh relevant caches automatically. To refresh remote transcripts,
  change `revision` under `[transcripts]` in the TOML profile. Old `work_v16`
  folders are preserved but not reused; current work uses `work_v17`.
- **No voice available:** cast all main characters and add guest voices.
- **Subtitle boxes:** install a CJK font such as Noto Sans CJK and restart.
- **Install errors:** confirm the environment uses Python 3.12. Keep the error
  text, but remove secrets and private paths before sharing it.
- **Output from another clip was replaced:** exports include an input-content identifier. A re-render of
  the same input replaces its previous exports; copy anything you want to keep.

## 7. Keep your work private

No output needs to be uploaded. The public project can contain only source code,
documentation and tests. Keep your recordings, transcripts, voice permissions,
keys, configuration and generated dubs on your own computer.

Read [the personal-use disclaimer](../DISCLAIMER.md). Private/educational use does
not automatically give permission to process copyrighted material or use a voice.
If you later decide to share an output, check the necessary rights and disclosure
requirements separately. The MP4 includes AI-dubbing metadata, but metadata alone
is not a guarantee of legal compliance.
