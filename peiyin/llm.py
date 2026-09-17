"""OpenAI / Gemini client plumbing.

A single place for JSON-mode chat completions, model selection and retries, so
`speakers` (casting) and `translate` can share it without importing each other.
"""

import json
import re
import time

import httpx

from .config import (
    GEMINI_BASE,
    GEMINI_MODEL,
    OPENAI_MODEL,
    OPENAI_URL,
    llm_model,
)


def http_post(url, params=None, headers=None, payload=None, timeout=120):
    last = None
    for attempt in range(4):
        try:
            with httpx.Client(timeout=timeout, trust_env=True) as c:
                r = c.post(url, params=params, headers=headers, json=payload)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 503):
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except httpx.HTTPError as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed after retries: {last}")


def llm_json(llm_key, system, user, max_retries=3):
    """One JSON-mode call to ChatGPT (sk-) or Gemini (anything else)."""
    for attempt in range(max_retries):
        try:
            if llm_key.startswith("sk-"):
                data = http_post(OPENAI_URL,
                                 headers={"Authorization": "Bearer " + llm_key},
                                 payload={"model": pick_openai_model(llm_key),
                                          "response_format": {"type": "json_object"},
                                          "messages": [
                                              {"role": "system", "content": system},
                                              {"role": "user", "content": user}]})
                raw = data["choices"][0]["message"]["content"]
            else:
                data = http_post(f"{GEMINI_BASE}/{llm_model() or GEMINI_MODEL}:generateContent",
                                 headers={"x-goog-api-key": llm_key},
                                 payload={"systemInstruction": {"parts": [{"text": system}]},
                                          "contents": [{"role": "user",
                                                        "parts": [{"text": user}]}],
                                          "generationConfig": {
                                              "responseMimeType": "application/json",
                                              "temperature": 0.4}})
                raw = "".join(p.get("text", "") for p in
                              data["candidates"][0]["content"]["parts"])
            raw = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"language model call failed: {last}")


def pick_openai_model(llm_key):
    """Use the same explicit/default model for casting and translation."""
    return llm_model() or OPENAI_MODEL


def llm_json_model(llm_key, system, user, model=None):
    """llm_json but with an explicit OpenAI model (Gemini path unchanged)."""
    if llm_key.startswith("sk-") and model:
        for attempt in range(3):
            try:
                data = http_post(OPENAI_URL,
                                 headers={"Authorization": "Bearer " + llm_key},
                                 payload={"model": model,
                                          "response_format": {"type": "json_object"},
                                          "messages": [
                                              {"role": "system", "content": system},
                                              {"role": "user", "content": user}]})
                raw = data["choices"][0]["message"]["content"]
                raw = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()
                return json.loads(raw)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"language model call failed: {last}")
    return llm_json(llm_key, system, user)
