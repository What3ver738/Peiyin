"""Regressions for private local use, cache boundaries and release packaging."""

import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from peiyin import config, llm, pipeline, profiles
from peiyin.cache import profile_fingerprint, script_workdir
from peiyin.util import file_key


def test_equal_size_same_named_files_do_not_share_cache(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first, second = a / "clip.mp4", b / "clip.mp4"
    first.write_bytes(b"aaaa")
    second.write_bytes(b"bbbb")
    assert file_key(first) != file_key(second)
    before = file_key(first)
    first.write_bytes(b"cccc")
    assert file_key(first) != before


def test_settings_failure_keeps_previous_file_and_cleans_temp(monkeypatch):
    config.save_config(fish_key="test-original")
    before = config.CONFIG_PATH.read_bytes()

    def fail_replace(src, dst):
        if os.name != "nt":
            assert stat.S_IMODE(Path(src).stat().st_mode) == 0o600
        raise OSError("disk unavailable")

    monkeypatch.setattr(config.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="Could not save"):
        config.save_config(fish_key="test-new")
    assert config.CONFIG_PATH.read_bytes() == before
    assert not list(config.CONFIG_PATH.parent.glob(".config-*"))


def test_corrupt_config_is_not_silently_overwritten():
    original = config.CONFIG_PATH.read_bytes()
    try:
        config.CONFIG_PATH.write_text("not json")
        with pytest.raises(RuntimeError, match="Cannot read"):
            config.save_config(output_dir="test")
        assert config.CONFIG_PATH.read_text() == "not json"
    finally:
        config.CONFIG_PATH.write_bytes(original)


def test_environment_keys_override_ui_without_persisting(monkeypatch):
    config.save_config(fish_key=None, llm_key=None)
    monkeypatch.setenv("FISH_API_KEY", "env-fish")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert config.resolve_keys("sk-ui", "ui-fish") == ("sk-env", "env-fish")
    assert "fish_key" not in config.load_config()
    assert "llm_key" not in config.load_config()


@pytest.mark.parametrize("chosen", [None, "custom-model"])
def test_openai_translation_and_casting_use_same_model(monkeypatch, chosen):
    config.save_config(llm_model=chosen)
    requests = []

    def post(url, **kwargs):
        requests.append(kwargs["payload"])
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    monkeypatch.setattr(llm, "http_post", post)
    llm.llm_json("sk-test", "JSON please", "test")
    llm.llm_json_model("sk-test", "JSON please", "test",
                       model=llm.pick_openai_model("sk-test"))
    assert [r["model"] for r in requests] == [chosen or config.OPENAI_MODEL] * 2
    assert all("temperature" not in r for r in requests)
    config.save_config(llm_model=None)


def test_mandarin_only_is_explicit():
    with pytest.raises(profiles.ProfileError, match="Mandarin"):
        profiles.parse_profile({"name": "Example", "target_language": "fr",
                                "characters": [{"name": "Alice"}]})


def test_packaged_profile_and_escaped_name():
    name = 'A "quoted" show\nwith a backslash \\'
    path = profiles.create_profile("escaped-name", show_name=name)
    assert profiles.load_profile(path).name == name
    assert (profiles.BUNDLED_PROFILE_DIR / "TEMPLATE.toml").exists()


def test_cache_changes_with_profile_model_provider_and_transcripts(tmp_path, monkeypatch):
    folder = tmp_path / "transcripts"
    folder.mkdir()
    f = folder / "0101.txt"
    f.write_text("Alice: first")
    prof = replace(profiles.active(), transcripts=profiles.Transcripts(local_folder=str(folder)))
    monkeypatch.setattr(profiles, "active", lambda: prof)
    config.save_config(llm_model=None)
    original = script_workdir(tmp_path, "small", "sk-test")
    assert original == script_workdir(tmp_path, "small", "sk-another-key")
    assert original != script_workdir(tmp_path, "medium", "sk-test")
    assert original != script_workdir(tmp_path, "small", "gemini-test")
    assert original != script_workdir(tmp_path, "small", "sk-test", False)
    f.write_text("Alice: other")
    assert original != script_workdir(tmp_path, "small", "sk-test")
    changed = script_workdir(tmp_path, "small", "sk-test")
    config.save_config(llm_model="custom-model")
    assert changed != script_workdir(tmp_path, "small", "sk-test")
    config.save_config(llm_model=None)
    assert profile_fingerprint(prof) != profile_fingerprint(replace(prof, name="Different"))
    assert profile_fingerprint(prof) != profile_fingerprint(
        replace(prof, transcripts=replace(prof.transcripts, revision="updated")))


def test_build_script_does_not_reuse_edits_after_model_change(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"synthetic input")
    config.save_config(llm_model=None)
    monkeypatch.setattr(pipeline, "extract_audio", lambda *args: ("fake.wav", "full.wav"))
    monkeypatch.setattr(pipeline, "transcribe", lambda *args, **kw: [
        {"start": 0., "end": 1., "en": "An original line"}])
    monkeypatch.setattr(pipeline, "cast_speakers", lambda *args, **kw:
                        (["Mara"], [], [], []))
    translated = []

    def translate(script, *args, **kwargs):
        translated.append(True)
        return [{**ln, "zh": "新的台词"} for ln in script]

    monkeypatch.setattr(pipeline, "translate_llm", translate)
    first, _ = pipeline.build_script(video, "sk-test", "small", tmp_path, False)
    stage = script_workdir(tmp_path, "small", "sk-test", False)
    first[0]["zh"] = "编辑"
    (stage / "script_edited.json").write_text(json.dumps(first))
    reused, edited = pipeline.build_script(video, "sk-test", "small", tmp_path, False)
    assert edited and reused[0]["zh"] == "编辑"
    assert len(translated) == 1
    config.save_config(llm_model="another-model")
    fresh, edited = pipeline.build_script(video, "sk-test", "small", tmp_path, False)
    assert not edited and fresh[0]["zh"] == "新的台词"
    assert len(translated) == 2
    config.save_config(llm_model=None)


def test_local_transcript_edits_change_parsed_database(tmp_path, monkeypatch):
    from peiyin.episodes import _cache_dir, ensure_scripts
    folder = tmp_path / "transcripts"
    folder.mkdir()
    f = folder / "0101.txt"
    f.write_text("\n".join(f"Mara: This is original line {i}." for i in range(6)))
    prof = replace(profiles.active(), transcripts=profiles.Transcripts(local_folder=str(folder)))
    monkeypatch.setattr(profiles, "active", lambda: prof)
    ensure_scripts()
    old = _cache_dir(prof)
    f.write_text(f.read_text().replace("original", "updated"))
    ensure_scripts()
    new = _cache_dir(prof)
    assert old != new
    assert "updated" in (new / "0101.json").read_text()


def test_ui_never_prefills_saved_credentials_and_serializes_changes(monkeypatch):
    from peiyin.ui import build_ui
    config.save_config(fish_key="private-saved-fish", llm_key="private-saved-llm")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    ui = build_ui()
    rendered = json.dumps(ui.config, default=str)
    assert "private-saved-fish" not in rendered
    assert "private-saved-llm" not in rendered
    assert all(f.concurrency_id == "peiyin-local" for f in ui.fns.values())
    assert any(f.name == "prepare_review" for f in ui.fns.values())
    assert any(f.name == "render_review" for f in ui.fns.values())
    config.save_config(fish_key=None, llm_key=None)


def test_render_pipeline_exports_and_reuses_tts_then_refreshes_model(
        tmp_path, monkeypatch, synthetic_profile, tone_wav):
    """Real FFmpeg/mixing; synthetic tones instead of paid voices or media."""
    import shutil

    from peiyin import tts
    from peiyin.util import run_ffmpeg

    source = tmp_path / "original.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=160x120:d=3",
                "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)])
    clip = tone_wav(tmp_path / "voice.wav", 330, dur=0.5)
    calls = []

    class SyntheticVoice:
        name = "fish"

        def synthesize(self, text, voice_id, path, **kwargs):
            calls.append((text, voice_id))
            shutil.copyfile(clip, path)

    config.save_config(output_dir=str(tmp_path / "exports"), fish_model="model-one")
    tts.save_voice_cast({"Alice": "synthetic-voice"})
    monkeypatch.setattr(pipeline, "ensure_fish_auto_pool", lambda key: None)
    monkeypatch.setattr(pipeline, "detect_speech_regions", lambda wav: ([(0.2, 1.2)], "synthetic"))
    script = [{"start": .2, "end": 1.2, "en": "Original", "zh": "你好", "speaker": "Alice"}]
    out, srt = pipeline.render_from_script(source, script, .3, False, provider=SyntheticVoice())
    assert Path(out).stat().st_size > 1000
    assert "你好" in Path(srt).read_text(encoding="utf-8")
    pipeline.render_from_script(source, script, .3, False, provider=SyntheticVoice())
    assert len(calls) == 1
    config.save_config(fish_model="model-two")
    pipeline.render_from_script(source, script, .3, False, provider=SyntheticVoice())
    assert len(calls) == 2
    config.save_config(fish_model=None, output_dir=None)


def test_review_callbacks_apply_edits_and_reject_changed_video(tmp_path, monkeypatch):
    from peiyin import ui
    config.save_config(llm_model=None)
    app = ui.build_ui()
    prepare = next(f.fn for f in app.fns.values() if f.name == "prepare_review")
    render = next(f.fn for f in app.fns.values() if f.name == "render_review")
    video = tmp_path / "test.mp4"
    video.write_bytes(b"video-a")
    monkeypatch.setattr(ui, "build_script", lambda *args, **kwargs: ([{
        "start": 0., "end": 1., "speaker": "Mara", "en": "original", "zh": "初稿"}], False))
    outputs = []

    def render_fake(video, script, *args, **kwargs):
        outputs.append(script)
        return str(video), str(video)

    monkeypatch.setattr(ui, "render_from_script", render_fake)
    args = ([str(video)], "sk-test", "Fast listening", False, "fish-test", str(tmp_path))
    state, rows, _ = prepare(*args, progress=lambda *a, **kw: None)
    Path(state["stage"]).mkdir(parents=True, exist_ok=True)
    rows[0][4] = "修改"
    render(state, rows, *args, .3, False, False, progress=lambda *a, **kw: None)
    assert outputs[0][0]["zh"] == "修改"
    assert json.loads((Path(state["stage"]) / "script_edited.json").read_text())[0]["zh"] == "修改"
    video.write_bytes(b"video-b")
    with pytest.raises(Exception, match="Prepare the script again"):
        render(state, rows, *args, .3, False, False, progress=lambda *a, **kw: None)
    config.save_config(output_dir=None)
