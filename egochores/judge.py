"""LLM label judge for the subtask and hand-caption tasks (Gemini by default; any (gold, pred) -> bool works).

The reference numbers use gemini-3.5-flash through Gemini's OpenAI-compatible endpoint with temperature 0, thinking off
(reasoning_effort "none") and 4096 max tokens; these are the exact settings below. Decisions are cached on disk by
(model, task, gold, pred).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from egochores.prompts import judge_prompt

DEFAULT_MODEL = "gemini-3.5-flash"
_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


class GeminiJudge:
    def __init__(
        self, model: str = DEFAULT_MODEL, cache_path: str | Path | None = "judge_cache.json", workers: int = 16
    ):
        self.model = model
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise RuntimeError("set GEMINI_API_KEY (or pass your own judge callable)")
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, bool] = (
            json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}
        )
        self.workers = workers

    def _key(self, task: str, gold: str, pred: str) -> str:
        return json.dumps([self.model, task, gold, pred])

    def _ask(self, task: str, gold: str, pred: str) -> bool:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": judge_prompt(task, gold, pred)}],
                "temperature": 0,
                "max_tokens": 4096,
                "reasoning_effort": "none",
            }
        ).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    _URL, data=body, headers={"content-type": "application/json", "authorization": f"Bearer {self.key}"}
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    text = json.loads(resp.read())["choices"][0]["message"]["content"]
                match = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0)).get("match")
                if isinstance(match, bool):
                    return match
                raise ValueError(f"no boolean match in {text!r}")
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError

    def decide(self, task: str, pairs: set[tuple[str, str]]) -> None:
        """Judge every pair not already cached (concurrently), then persist the cache."""
        todo = [
            p for p in pairs if self._key(task, *p) not in self.cache and p[0].strip().lower() != p[1].strip().lower()
        ]
        with ThreadPoolExecutor(self.workers) as ex:
            for (g, p), m in zip(todo, ex.map(lambda gp: self._ask(task, *gp), todo)):
                self.cache[self._key(task, g, p)] = m
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache))

    def accept(self, task: str):
        def accept(gold: str, pred: str) -> bool:
            key = self._key(task, gold, pred)
            if key not in self.cache:
                self.cache[key] = self._ask(task, gold, pred)
            return self.cache[key]

        return accept
