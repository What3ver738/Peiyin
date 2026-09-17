"""The Gradio web interface."""

import json
import re
import shutil
import tempfile
import time
import traceback
import uuid
from pathlib import Path

from . import config, profiles, tts
from .cache import script_workdir
from .config import (
    APP_DIR,
    CONFIG_PATH,
    DIAG,
    PRODUCT_VERSION,
    WORK_ROOT,
    current_out_dir,
    load_config,
    save_config,
)
from .llm import llm_json
from .pipeline import build_script, render_from_script
from .tts import (
    build_fish_voice_map,
    check_fish,
    load_fish_auto_pool,
    load_fish_guest_pool,
)
from .util import file_key, reveal_folder


def check_keys(llm_key="", fish_key=""):
    out = []
    llm_key, fish_key = config.resolve_keys(llm_key, fish_key)
    if fish_key:
        out.append(check_fish(fish_key))
    else:
        out.append("No Fish Audio key given (needed for the voices).")
    if llm_key:
        try:
            llm_json(llm_key, "Reply with JSON only.",
                     'Return {"ok": true} exactly.')
            which = "CHATGPT" if llm_key.startswith("sk-") else "GEMINI"
            out.append(f"{which} (casting + translation): works.")
        except Exception as e:  # noqa: BLE001
            out.append(f"CHATGPT/GEMINI: FAILED -> {str(e)[:200]}")
    else:
        out.append("No ChatGPT/Gemini key: speaker casting and translation are "
                   "unavailable.")
    return "\n".join(out)


def script_to_rows(script):
    return [[i, f"{ln['start']:.2f}", ln["speaker"], ln["en"], ln.get("zh", "")]
            for i, ln in enumerate(script)]


def rows_to_script(rows, script):
    if rows is None:
        return script
    fixed = {s.lower(): s for s in profiles.labels()}
    fixed.update({ln["speaker"].lower(): ln["speaker"] for ln in script})
    for row in rows:
        if len(row) != 5:
            raise ValueError("The review table must keep its five columns.")
        try:
            i = int(row[0])
        except (TypeError, ValueError):
            continue
        if not (0 <= i < len(script)):
            continue
        sp = str(row[2]).strip()
        script[i]["speaker"] = fixed.get(sp.lower(), script[i]["speaker"])
        zh = str(row[4]).strip()
        if zh:
            script[i]["zh"] = zh
        elif script[i]["speaker"] == "Skip":
            script[i]["zh"] = ""
    return script


def build_ui():
    import gradio as gr

    cfg = load_config()

    def _profile_choices():
        return [f"{disp}  [{pid}]" for pid, disp in profiles.list_profiles()]

    def _pid_from_choice(choice):
        m = re.search(r"\[([^\]]+)\]\s*$", choice or "")
        return m.group(1) if m else ""

    def _profile_status():
        """One line describing the active profile and whether it has transcripts."""
        prof = profiles.active()
        who = ", ".join(prof.main_cast[:6]) or "(no characters)"
        if prof.has_transcripts:
            return (f"**{prof.name}** — {len(prof.main_cast)} main characters "
                    f"({who}). Transcripts: configured.")
        return (f"**{prof.name}** — {len(prof.main_cast)} main characters "
                f"({who}).\n\n"
                "⚠️ **No transcript source set for this profile.** Peiyin will "
                "group speakers by voice alone: it cannot identify the episode, "
                "and speaker accuracy is noticeably lower. Add a transcript "
                "source to the profile file to fix this — see "
                "`peiyin/data/profiles/TEMPLATE.toml`.")

    def pick_profile(choice):
        pid = _pid_from_choice(choice)
        try:
            profiles.set_active_profile(pid)
        except profiles.ProfileError as e:
            raise gr.Error(str(e))
        return _profile_status()

    def make_profile(new_name):
        if not (new_name or "").strip():
            raise gr.Error("Give the new profile a name first.")
        try:
            path = profiles.create_profile(new_name, show_name=new_name.strip())
        except profiles.ProfileError as e:
            raise gr.Error(str(e))
        profiles.active.cache_clear()
        return (gr.update(choices=_profile_choices()),
                f"Created `{path}`.\n\nOpen it in a text editor to add the "
                f"characters and your transcript source, then pick it above.")

    # ----------------------------------------------------------- Settings
    def test_fish_key(key):
        _, key = config.resolve_keys(fish=key)
        if not key:
            return "❌ No Fish Audio key given."
        return check_fish(key)

    def test_llm_key(key):
        key, _ = config.resolve_keys(llm=key)
        if not key:
            return "❌ No ChatGPT or Gemini key given."
        which = "ChatGPT" if key.startswith("sk-") else "Gemini"
        try:
            llm_json(key, "Reply with JSON only.", 'Return {"ok": true} exactly.')
            return f"✅ {which}: works."
        except Exception as e:  # noqa: BLE001
            return f"❌ {which}: {str(e)[:200]}"

    def save_settings(fish, llm, fish_m, llm_m, out_dir):
        save_config(fish_key=(fish or "").strip(),
                    llm_key=(llm or "").strip(),
                    fish_model=(fish_m or "").strip() or None,
                    llm_model=(llm_m or "").strip() or None,
                    output_dir=(out_dir or "").strip() or None)
        return (f"Saved to `{CONFIG_PATH}`. Keys are plaintext; keep your user "
                f"account private. Output folder: `{current_out_dir()}`.")

    def setup_state():
        """What still has to happen before a dub can run."""
        missing = []
        if not config.api_key("fish"):
            missing.append("a **Fish Audio key** (the voices)")
        if not config.llm_key():
            missing.append("an **OpenAI or Gemini key** (casting and translation)")
        if not tts.load_voice_cast():
            missing.append("a **voice for each character** (Cast tab)")
        if not missing:
            return ""
        return ("### Before your first dub\n\nYou still need "
                + "; ".join(missing) + ".\n\n"
                "Keys go in the **Settings** tab, or in environment variables "
                "(`FISH_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) or a "
                "`.env` file — see `.env.example`.")

    # ------------------------------------------------------------- Cast tab
    def cast_rows():
        """[[character, voice id], ...] for the active profile, main cast first."""
        cast = tts.load_voice_cast()
        return [[name, cast.get(name, "")] for name in profiles.main_cast()]

    def cast_characters():
        return profiles.main_cast()

    def save_cast(rows):
        """Persist the cast table. Stored per profile, in the config dir."""
        cast = {}
        for row in rows or []:
            if not row:
                continue
            name = str(row[0]).strip()
            vid = str(row[1]).strip() if len(row) > 1 and row[1] else ""
            if name:
                cast[name] = vid
        tts.save_voice_cast(cast)
        missing = [n for n in profiles.main_cast() if not cast.get(n)]
        if missing:
            return ("Saved. Still uncast: " + ", ".join(missing)
                    + ". A dub stops rather than borrow someone else's voice, "
                      "so cast everyone before dubbing.")
        return f"Saved. All {len(cast)} main characters are cast."

    def search_voices(query, fish_key, only_target_language):
        _, fish_key = config.resolve_keys(fish=fish_key)
        if not fish_key:
            raise gr.Error("Add your Fish Audio API key in Settings first.")
        lang = profiles.target_language() if only_target_language else ""
        try:
            hits = tts.search_fish_voices((fish_key or "").strip(),
                                          query=(query or "").strip(),
                                          language=lang)
        except RuntimeError as e:
            raise gr.Error(str(e))
        if not hits:
            return [["", "no matches — try a different search", "", ""]]
        return [[h["id"], h["title"], h["gender"], h["languages"]] for h in hits]

    def preview_cast_voice(character, rows, fish_key):
        _, fish_key = config.resolve_keys(fish=fish_key)
        vid = ""
        for row in rows or []:
            if row and str(row[0]).strip() == character:
                vid = str(row[1]).strip() if len(row) > 1 and row[1] else ""
                break
        if not vid:
            raise gr.Error(f"No voice id set for {character} yet.")
        try:
            return stage_download(tts.preview_voice(fish_key, vid,
                                     language=profiles.target_language()))
        except RuntimeError as e:
            raise gr.Error(str(e))

    def preview_any_voice(vid, fish_key):
        _, fish_key = config.resolve_keys(fish=fish_key)
        try:
            return stage_download(tts.preview_voice(fish_key, vid or "",
                                     language=profiles.target_language()))
        except RuntimeError as e:
            raise gr.Error(str(e))

    def save_guests(male_text, female_text, auto_on):
        """Guest pools are stored in the same 'M <id>' / 'F <id>' form the
        rest of the app already reads."""
        lines = []
        for g, text in (("M", male_text), ("F", female_text)):
            for raw in (text or "").splitlines():
                raw = raw.strip()
                if raw:
                    lines.append(f"{g} {raw}")
        save_config(fish_guest_pool_text="\n".join(lines) or None)
        tts.set_auto_public_voices(bool(auto_on))
        nm = sum(1 for ln in lines if ln.startswith("M "))
        nf = sum(1 for ln in lines if ln.startswith("F "))
        note = (" Public Fish voices are ALSO enabled for guests."
                if auto_on else
                " Public Fish voices are off; guests come only from this pool.")
        if not lines and not auto_on:
            return ("Saved, but the guest pool is empty and public voices are "
                    "off — any line spoken by someone outside the main cast "
                    "will stop the dub.")
        return f"Saved {nm} male and {nf} female guest voices.{note}"

    def _guest_text():
        pool = tts.load_fish_guest_pool()
        m = "\n".join(v for g, v in pool if g == "MALE")
        f = "\n".join(v for g, v in pool if g == "FEMALE")
        return m, f

    def refresh_cast_tab():
        m, f = _guest_text()
        return (cast_rows(),
                gr.update(choices=cast_characters(),
                          value=(cast_characters() or [None])[0]),
                m, f, tts.auto_public_voices_enabled())

    def show_voices():
        vm = build_fish_voice_map()
        pool = load_fish_guest_pool()
        nm = sum(1 for g, _ in pool if g == "MALE")
        nf = sum(1 for g, _ in pool if g == "FEMALE")
        am, af = load_fish_auto_pool()
        lines = ["Main characters (from your cast):"]
        lines += [f"  {r}: {vm.get(r) or '(not cast yet)'}"
                  for r in profiles.main_cast()]
        lines += ["", "Generic guest buckets:"]
        lines += [f"  {r}: {vm.get(r) or '(none available)'}"
                  for r in ("OtherMale", "OtherFemale")]
        lines += ["", f"Guest pool you pasted: {nm} male, {nf} female.",
                  f"Public voices auto-fetched: {len(am)} male, {len(af)} "
                  f"female (used for guests; never a cast voice)."]
        gv = tts.profile_setting("guest_voices") or {}
        if gv:
            lines += ["", "Named guests assigned so far:"]
            lines += [f"  {n}: {v}" for n, v in sorted(gv.items())]
        return "\n".join(lines)

    def reset_voices():
        """Clear per-guest assignments and the fetched public pool.

        The cast itself is untouched -- those are the user's own choices.
        """
        tts.save_profile_setting(guest_voices={})
        save_config(fish_auto_guest_pool=None)
        return ("Cleared this profile's guest voice assignments and the "
                "fetched public pool. Your cast main characters are "
                "untouched. Guests are reassigned on the next dub.")

    def _resolve(llm_key, model_label, fish_key, out_dir_text=""):
        llm_key, fish_key = config.resolve_keys(llm_key, fish_key)
        if not fish_key:
            raise gr.Error("Add your Fish Audio API key in Settings or the environment.")
        if not llm_key:
            raise gr.Error("Add your OpenAI or Gemini key in Settings or the environment.")
        # Only an explicit Settings save persists credentials.
        save_config(output_dir=(out_dir_text or "").strip() or None)
        profiles.active.cache_clear()
        profiles.active()  # reload external TOML edits before choosing cache keys
        model_size = "medium" if "accurate" in model_label else "small"
        return llm_key, model_size, fish_key

    def dub_one(video, llm_key, model_size, bg_volume, burn_subs,
                use_scripts, report, fish_key="", use_emotion=True):
        """Dub a single file. `report(frac, desc)` drives the progress bar.
        Returns (out_mp4, srt, status_text)."""
        workdir = WORK_ROOT / file_key(Path(video))
        workdir.mkdir(parents=True, exist_ok=True)
        DIAG.clear()
        script, _ = build_script(Path(video), llm_key, model_size, workdir,
                                 bool(use_scripts),
                                 lambda f, desc="": report(0.02 + 0.38 * f, desc))
        cast = {}
        for ln in script:
            cast[ln["speaker"]] = cast.get(ln["speaker"], 0) + 1
        out_mp4, srt = render_from_script(
            video, script, bg_volume, bool(burn_subs),
            lambda f, desc="": report(0.40 + 0.60 * f, desc), fish_key=fish_key,
            use_emotion=use_emotion)
        return stage_download(out_mp4), stage_download(srt), "Done.\n" + "\n".join(DIAG)

    def dub(video, llm_key, model_label, bg_volume, burn_subs, use_scripts,
            fish_key, emote, out_dir_text, progress=gr.Progress()):
        if not video:
            raise gr.Error("Drop a video file in first.")
        llm_key, model_size, fish_key = _resolve(
            llm_key, model_label, fish_key, out_dir_text)
        try:
            result = dub_one(video, llm_key, model_size,
                             bg_volume, burn_subs, use_scripts,
                             lambda f, desc="": progress(f, desc=desc),
                             fish_key=fish_key, use_emotion=bool(emote))
            reveal_folder(current_out_dir())      # pop the folder open when done
            return result
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise gr.Error(f"Something went wrong: {e}")

    def dub_batch(videos, llm_key, model_label, bg_volume, burn_subs,
                  use_scripts, fish_key, emote, out_dir_text,
                  progress=gr.Progress()):
        """Work through a whole queue of episodes one after another.
        Yields after each so the page shows live progress; safe to leave for
        hours. One failure never stops the queue."""
        if not videos:
            raise gr.Error("Add one or more video files to the queue first.")
        if isinstance(videos, str):
            videos = [videos]
        paths = [v["name"] if isinstance(v, dict) else
                 getattr(v, "name", v) for v in videos]
        llm_key, model_size, fish_key = _resolve(
            llm_key, model_label, fish_key, out_dir_text)

        total = len(paths)
        done_files, log_lines = [], []
        start = time.time()
        for n, path in enumerate(paths, 1):
            name = Path(path).name
            log_lines.append(f"[{n}/{total}] {name} — working...")
            yield done_files, "\n".join(log_lines)

            def report(f, desc="", _n=n, _name=name):
                overall = ((_n - 1) + max(0.0, min(f, 1.0))) / total
                progress(overall, desc=f"[{_n}/{total}] {_name}: {desc}")

            try:
                out_mp4, srt, status = dub_one(
                    path, llm_key, model_size, bg_volume, burn_subs,
                    use_scripts, report, fish_key=fish_key,
                    use_emotion=bool(emote))
                done_files = done_files + [out_mp4, srt]
                ep = next((ln.split("Episode identified:")[1].split("(")[0].strip()
                           for ln in status.splitlines()
                           if "Episode identified:" in ln), "")
                warn = " ⚠️ check status" if "WARNING" in status else ""
                log_lines[-1] = (f"[{n}/{total}] {name} — done"
                                 + (f" ({ep})" if ep else "") + warn)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                log_lines[-1] = f"[{n}/{total}] {name} — FAILED: {str(e)[:80]}"
            elapsed = time.time() - start
            eta = elapsed / n * (total - n)
            log_lines.append(f"    {n}/{total} finished · about "
                             f"{eta/60:.0f} min left for the rest")
            yield done_files, "\n".join(log_lines)
            log_lines.pop()   # drop the transient ETA line before next episode

        log_lines.append(f"All {total} episodes finished in "
                         f"{(time.time()-start)/60:.0f} min. Saved to: "
                         f"{current_out_dir()}")
        if done_files:
            reveal_folder(current_out_dir())      # pop the folder open when done
        yield done_files, "\n".join(log_lines)

    def prepare_review(videos, key, model_label, use_scripts, fish, out_dir,
                       progress=gr.Progress()):
        video = _first(videos)
        key, model_size, _ = _resolve(key, model_label, fish, out_dir)
        work = WORK_ROOT / file_key(video)
        work.mkdir(parents=True, exist_ok=True)
        DIAG.clear()
        script, _ = build_script(video, key, model_size, work, bool(use_scripts), progress)
        stage = script_workdir(work, model_size, key, bool(use_scripts))
        state = {"video": video, "identity": file_key(video),
                 "stage": str(stage), "script": script}
        return state, script_to_rows(script), "Ready: edit Speaker or Mandarin, then render."

    def render_review(state, rows, videos, key, model_label, use_scripts, fish,
                      out_dir, bg_volume, burn_subs, emote, progress=gr.Progress()):
        if not state:
            raise gr.Error("Prepare a script first.")
        video = _first(videos)
        key, model_size, fish = _resolve(key, model_label, fish, out_dir)
        identity = file_key(video)
        stage = script_workdir(WORK_ROOT / identity, model_size, key, bool(use_scripts))
        if identity != state["identity"] or str(stage) != state["stage"]:
            raise gr.Error("Video, profile or settings changed. Prepare the script again.")
        script = rows_to_script(rows, [dict(ln) for ln in state["script"]])
        (stage / "script_edited.json").write_text(
            json.dumps(script, ensure_ascii=False), encoding="utf-8")
        out_mp4, srt = render_from_script(video, script, bg_volume, bool(burn_subs),
                                         progress, fish_key=fish, use_emotion=bool(emote))
        return stage_download(out_mp4), stage_download(srt), "Done.\n" + "\n".join(DIAG)

    with gr.Blocks(title="Peiyin 配音", analytics_enabled=False,
                   delete_cache=(3600, 86400)) as demo:
        gr.Markdown(f"# 🎬 Peiyin 配音 · v{PRODUCT_VERSION}\n"
                    "*AI dubbing for private language learning.* Use a video you "
                    "have permission to process. Peiyin works out who says each line, "
                    "translates it, and re-voices it — automatically.")
        gr.Markdown(
            "**Personal-use tool.** Use only media, transcripts and voices you have "
            "permission to process. Personal or educational use does not automatically "
            "grant rights. Do not impersonate people or share outputs without the "
            "necessary rights. Dialogue is sent to your LLM provider; translated "
            "text is sent to Fish Audio. Outputs are AI-generated and may be wrong.")
        setup_banner = gr.Markdown(setup_state())   # refreshed below
        with gr.Tab("Dub"):
          with gr.Row():
            with gr.Column():
                _choices = _profile_choices()
                _cur = profiles.active_profile_id()
                profile_pick = gr.Dropdown(
                    _choices,
                    value=next((c for c in _choices
                                if _pid_from_choice(c) == _cur), None)
                          or (_choices[0] if _choices else None),
                    label="Show profile — who the characters are and where your "
                          "transcripts come from")
                profile_info = gr.Markdown(_profile_status())
                with gr.Accordion("Create a new profile", open=False):
                    new_profile_name = gr.Textbox(
                        label="Show name",
                        placeholder="My show",
                        info="Writes a filled-in copy of TEMPLATE.toml into "
                             "your config folder for you to edit.")
                    make_profile_btn = gr.Button("Create profile", size="sm")
                video_in = gr.File(
                    label="Your episodes — add one or many (mp4 / mkv / avi)",
                    file_types=["video"], type="filepath",
                    file_count="multiple")
                use_scripts = gr.Checkbox(
                    value=True,
                    label="Use transcripts to know exactly who speaks each "
                          "line (recommended)")
                fish_key = gr.Textbox(label="Fish Audio API key (voices)",
                                      value="",
                                      type="password")
                llm_key = gr.Textbox(label="ChatGPT key (sk-...) or Gemini key",
                                     value="", type="password")
                with gr.Accordion("Advanced", open=False):
                    model = gr.Radio(["Fast listening",
                                      "More accurate listening (slower)"],
                                     value="Fast listening",
                                     label="English transcription")
                    bg = gr.Slider(0.0, 0.6, value=0.30, step=0.02,
                                   label="Laugh track between lines "
                                         "(auto-ducks under speech)")
                    burn = gr.Checkbox(value=True,
                                       label="Chinese subtitles in the picture")
                    emote = gr.Checkbox(
                        value=True,
                        label="Act out emotion in the voice (Fish) — the cue is "
                              "heard, never shown or spoken. Turn off for flat "
                              "delivery.")
                    gr.Markdown("*Voices are chosen in the **Cast** tab.*")
                    out_dir_box = gr.Textbox(
                        label="Output folder — finished episodes (.mp4 + .srt) "
                              "are saved here automatically and the folder opens "
                              "when done. Default: ~/Peiyin Output. Point it at "
                              "an external drive for a whole series if you like.",
                        value=str(current_out_dir()))
                    voice_box = gr.Textbox(label="Locked character voices",
                                           lines=8, interactive=False)
                    with gr.Row():
                        show_btn = gr.Button("Show voices", size="sm")
                        reset_btn = gr.Button("Reset voices", size="sm")
                check_btn = gr.Button("Check my keys 🔑", size="sm")
                dub_btn = gr.Button("Dub it 🎬 (one episode)",
                                    variant="secondary")
                batch_btn = gr.Button("Dub the whole queue 🎬🎬",
                                      variant="primary", size="lg")
                gr.Markdown("*The queue runs episode by episode and can be left "
                            "for hours. Each finished episode is saved to your "
                            "Output folder (see Advanced) as soon as it's done — "
                            "by default **~/Peiyin Output**, which opens "
                            "automatically when it finishes.*")
            with gr.Column():
                video_out = gr.File(label="Dubbed episode")
                srt_out = gr.File(label="Chinese subtitles (.srt)")
                batch_out = gr.File(label="Finished files (whole queue)",
                                    file_count="multiple")
                log = gr.Textbox(label="Status / queue progress", lines=14)

        with gr.Tab("Review script"):
            gr.Markdown("Optional: prepare the first queued video, edit the speaker or "
                        "Mandarin text, then render. Use **Skip** to mute a line. "
                        "Preparing calls the translation API; rendering calls Fish Audio.")
            review_state = gr.State()
            prepare_btn = gr.Button("Prepare script", variant="primary")
            review_table = gr.Dataframe(
                headers=["Line", "Start (s)", "Speaker", "English", "Mandarin"],
                datatype=["number", "str", "str", "str", "str"], type="array",
                column_count=5,
                static_columns=[0, 1, 3], interactive=True, label="Script")
            review_status = gr.Markdown()
            render_btn = gr.Button("Render reviewed script", variant="primary")

        with gr.Tab("Cast") as cast_tab:
            gr.Markdown(
                "### Choose a voice for every character\n"
                "Peiyin ships **no voices**. Paste a Fish Audio voice id for "
                "each main character, or search the public library below.\n\n"
                "⚠️ **You are responsible for the voices you use.** Only use a "
                "voice you have the right to use, and never clone a real "
                "person without their consent. Peiyin cannot tell what a "
                "public voice is a clone of.\n\n"
                "Casting is saved per profile, in your config folder — never "
                "in the project.")
            _m0, _f0 = _guest_text()
            cast_table = gr.Dataframe(
                headers=["Character", "Fish voice id"],
                datatype=["str", "str"], column_count=2, type="array",
                value=cast_rows(), interactive=True,
                label="Main characters (this profile)")
            with gr.Row():
                save_cast_btn = gr.Button("Save cast", variant="primary")
                preview_pick = gr.Dropdown(
                    cast_characters(),
                    value=(cast_characters() or [None])[0],
                    label="Preview character", scale=2)
                preview_cast_btn = gr.Button("▶ Preview")
            cast_status = gr.Markdown()
            preview_audio = gr.Audio(label="Preview", type="filepath")

            with gr.Accordion("Search Fish voices", open=False):
                with gr.Row():
                    search_q = gr.Textbox(
                        label="Search", scale=3,
                        placeholder="e.g. 男声 / narrator / 温柔")
                    only_lang = gr.Checkbox(
                        value=True,
                        label=f"Only {profiles.target_language()}")
                    search_btn = gr.Button("Search", scale=1)
                results = gr.Dataframe(
                    headers=["Voice id", "Title", "Gender?", "Languages"],
                    datatype=["str", "str", "str", "str"],
                    column_count=4, type="array", interactive=False,
                    label="Copy an id into the cast table above")
                with gr.Row():
                    try_id = gr.Textbox(label="Preview any voice id", scale=3)
                    try_btn = gr.Button("▶ Preview", scale=1)

            with gr.Accordion("Guest voices", open=False):
                gr.Markdown(
                    "Anyone outside the main cast draws from these pools, "
                    "gender-matched where possible. A guest is **never** given "
                    "a main character's voice.")
                with gr.Row():
                    guest_m = gr.Textbox(label="Male guest voices — one id "
                                               "per line", lines=5, value=_m0)
                    guest_f = gr.Textbox(label="Female guest voices — one id "
                                               "per line", lines=5, value=_f0)
                auto_pool = gr.Checkbox(
                    value=tts.auto_public_voices_enabled(),
                    label="Also draw guest voices from Fish's public library")
                gr.Markdown(
                    "*Off by default. The public library is user-uploaded and "
                    "can contain unauthorised clones of real people; Peiyin "
                    "cannot verify what a voice is a clone of. Turn this on "
                    "only if you accept responsibility for what it returns.*")
                save_guests_btn = gr.Button("Save guest voices",
                                            variant="primary")
                guest_status = gr.Markdown()

        with gr.Tab("Settings") as settings_tab:
            gr.Markdown(
                "### Keys and models\n"
                "Both services are paid; what a run costs depends on your usage "
                "and their pricing. Keys saved here are plaintext in your config "
                "folder (private file permissions on macOS/Linux; your account's "
                "folder permissions on Windows). Saved keys are never prefilled "
                "in the browser; blank fields keep existing keys.\n\n"
                "You can skip this tab entirely by setting `FISH_API_KEY` and "
                "`OPENAI_API_KEY` or `GEMINI_API_KEY` in your environment, or "
                "in a `.env` file. **An environment variable always wins over "
                "a stored key**, so nothing need be written to disk.")
            with gr.Row():
                set_fish = gr.Textbox(
                    label="Fish Audio API key", type="password",
                    value="",
                    info=f"currently from: {config.key_source('fish')}")
                test_fish_btn = gr.Button("Test key", scale=0)
            fish_result = gr.Markdown()
            with gr.Row():
                set_llm = gr.Textbox(
                    label="ChatGPT key (sk-...) or Gemini key", type="password",
                    value="",
                    info="currently from: " + (
                        config.key_source("openai")
                        if config.api_key("openai") else
                        config.key_source("gemini")))
                test_llm_btn = gr.Button("Test key", scale=0)
            llm_result = gr.Markdown()
            with gr.Accordion("Models and output folder", open=False):
                set_fish_model = gr.Textbox(
                    label="Fish Audio model",
                    value=cfg.get("fish_model", ""),
                    placeholder=config.FISH_TTS_MODEL,
                    info="Blank uses the default. The default supports "
                         "[bracket] delivery cues.")
                set_llm_model = gr.Textbox(
                    label="Casting / translation model",
                    value=cfg.get("llm_model", ""),
                    placeholder="blank = provider default",
                    info=f"Used for both casting and translation. Defaults: "
                         f"{config.OPENAI_MODEL} / {config.GEMINI_MODEL}. "
                         "Key tests and voice previews may incur API charges.")
                set_out_dir = gr.Textbox(
                    label="Output folder", value=str(current_out_dir()))
            save_settings_btn = gr.Button("Save settings", variant="primary")
            settings_result = gr.Markdown()

        test_fish_btn.click(test_fish_key, [set_fish], [fish_result])
        test_llm_btn.click(test_llm_key, [set_llm], [llm_result])
        save_settings_btn.click(
            save_settings,
            [set_fish, set_llm, set_fish_model, set_llm_model, set_out_dir],
            [settings_result]).then(setup_state, None, [setup_banner]).then(
                lambda: str(current_out_dir()), None, [out_dir_box])
        settings_tab.select(
            lambda: (gr.update(info=f"currently from: {config.key_source('fish')}"),
                     gr.update(info="currently from: " + (
                         config.key_source("openai") if config.api_key("openai")
                         else config.key_source("gemini")))),
            None, [set_fish, set_llm])

        cast_tab.select(refresh_cast_tab, None,
                        [cast_table, preview_pick, guest_m, guest_f, auto_pool])
        save_cast_btn.click(save_cast, [cast_table], [cast_status]).then(
            setup_state, None, [setup_banner])
        preview_cast_btn.click(preview_cast_voice,
                               [preview_pick, cast_table, fish_key],
                               [preview_audio])
        search_btn.click(search_voices, [search_q, fish_key, only_lang],
                         [results])
        try_btn.click(preview_any_voice, [try_id, fish_key], [preview_audio])
        save_guests_btn.click(save_guests, [guest_m, guest_f, auto_pool],
                              [guest_status])
        profile_pick.change(pick_profile, [profile_pick], [profile_info]).then(
            refresh_cast_tab, None,
            [cast_table, preview_pick, guest_m, guest_f, auto_pool])
        make_profile_btn.click(make_profile, [new_profile_name],
                               [profile_pick, profile_info])
        check_btn.click(check_keys, [llm_key, fish_key], [log])
        def _first(files):
            # the single-episode button takes the first queued file
            if not files:
                raise gr.Error("Add a video file first.")
            f = files[0] if isinstance(files, list) else files
            return f["name"] if isinstance(f, dict) else getattr(f, "name", f)

        dub_btn.click(
            lambda v, *a: dub(_first(v), *a),
            [video_in, llm_key, model, bg, burn, use_scripts, fish_key,
             emote, out_dir_box],
            [video_out, srt_out, log])
        prepare_btn.click(prepare_review,
                          [video_in, llm_key, model, use_scripts, fish_key, out_dir_box],
                          [review_state, review_table, review_status])
        render_btn.click(render_review,
                         [review_state, review_table, video_in, llm_key, model,
                          use_scripts, fish_key, out_dir_box, bg, burn, emote],
                         [video_out, srt_out, log])
        batch_btn.click(
            dub_batch,
            [video_in, llm_key, model, bg, burn, use_scripts, fish_key,
             emote, out_dir_box],
            [batch_out, log])
        show_btn.click(show_voices, None, [voice_box])
        reset_btn.click(reset_voices, None, [voice_box])
    # Settings/profile changes cannot interleave with a paid render in another tab.
    for event in demo.fns.values():
        event.concurrency_id = "peiyin-local"
        event.concurrency_limit = 1
    demo.queue(default_concurrency_limit=1, max_size=16)
    return demo


# Only explicit returned artifacts are copied here. Never allow a whole user
# output folder or the private working/config directory through the file route.
_DOWNLOADS = tempfile.TemporaryDirectory(prefix="peiyin-downloads-")


def stage_download(path):
    src = Path(path)
    dest = Path(_DOWNLOADS.name) / uuid.uuid4().hex / src.name
    dest.parent.mkdir(mode=0o700)
    shutil.copyfile(src, dest)
    return str(dest)


def launch_ui(open_browser=True, *, prevent_thread_lock=False):
    import gradio as gr
    demo = build_ui()
    return demo.launch(inbrowser=open_browser, show_error=False,
                       prevent_thread_lock=prevent_thread_lock,
                       server_name="127.0.0.1", share=False,
                       allowed_paths=[_DOWNLOADS.name],
                       blocked_paths=[str(APP_DIR), str(Path.cwd() / ".env")],
                       strict_cors=True, enable_monitoring=False, mcp_server=False,
                       run_history=False, footer_links=[],
                       css=".gradio-container{max-width:900px !important;margin:auto;}",
                       theme=gr.themes.Soft(primary_hue="red"))
