"""A tiny persistent cache for LLM responses.

Identical game states recur constantly across tournament repetitions and across
strategies that produce the same histories, so caching turns thousands of API
calls into hundreds. Keyed on (model, system_prompt, temperature, user_prompt).

Stored as JSON on disk so a re-run picks up where the last left off. Thread-safe
enough for Axelrod's default serial play; if you enable parallel tournaments,
run with a single process or switch to a sqlite backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Optional, Tuple

Key = Tuple[str, str, float, str]


class ResponseCache:
    def __init__(self, path: str = "results/data/llm_cache.json") -> None:
        self.path = path
        self._lock = threading.Lock()
        self._store: dict[str, str] = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as fh:
                    self._store = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self._store = {}

    @staticmethod
    def _hash(key: Key) -> str:
        model, system, temp, user = key
        blob = json.dumps([model, system, temp, user], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: Key) -> Optional[str]:
        with self._lock:
            return self._store.get(self._hash(key))

    def set(self, key: Key, value: str) -> None:
        with self._lock:
            self._store[self._hash(key)] = value

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self._store, fh)
            os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._store)

    # The threading.Lock is not picklable; exclude it so the cache can survive
    # being passed through Axelrod's machinery, and restore it on unpickle.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
