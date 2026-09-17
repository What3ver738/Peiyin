"""Exercise the real local HTTP interface with invented files and no API keys."""

import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

# Import from a checkout when invoked as `python scripts/smoke_ui.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix="peiyin-http-check-") as root:
        os.environ["PEIYIN_HOME"] = root
        os.environ["GRADIO_TEMP_DIR"] = str(Path(root) / "uploads")
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"  # launch must override this
        os.environ["MPLCONFIGDIR"] = str(Path(root) / "mpl")
        for key in ("FISH_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
            os.environ.pop(key, None)
        # Keep a checkout .env out of this synthetic, key-free check.
        previous_cwd = Path.cwd()
        os.chdir(root)
        import httpx

        from peiyin import config, ui

        config.save_config(fish_key="synthetic-private-secret")
        download = Path(root) / "result.txt"
        download.write_text("synthetic-result")
        staged = ui.stage_download(download)
        private_work = config.WORK_ROOT / "private.txt"
        private_work.write_text("synthetic-private-work")
        app, url, share = ui.launch_ui(False, prevent_thread_lock=True)
        try:
            assert url.startswith("http://127.0.0.1:") and not share
            with httpx.Client(base_url=url, trust_env=False, timeout=15) as client:
                assert client.get("/").status_code == 200
                response = client.get("/config")
                assert response.status_code == 200
                assert "synthetic-private-secret" not in response.text
                response = client.get("/gradio_api/file=" + quote(staged, safe="/"))
                assert response.status_code == 200 and response.text == "synthetic-result"
                for denied in (config.CONFIG_PATH, private_work, download):
                    response = client.get("/gradio_api/file=" + quote(str(denied), safe="/"))
                    assert response.status_code in (403, 404), (denied.name, response.status_code)
                response = client.post("/gradio_api/upload", files={
                    "files": ("synthetic.mp4", b"synthetic-upload", "video/mp4")})
                assert response.status_code == 200, response.text[:100]
                uploaded = Path(response.json()[0])
                assert uploaded.read_bytes() == b"synthetic-upload"
            print("PASS: loopback only; UI/config/upload/download work; secrets and private files denied.")
        finally:
            # launch returns the ASGI app; the enclosing Blocks owns the server.
            app.get_blocks().close()
            os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
