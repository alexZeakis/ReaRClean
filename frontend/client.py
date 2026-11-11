"""Client for the ReaRClean backend.

The frontend holds no cleaning logic of its own; everything goes through here.
"""

import math
import requests
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE = os.environ.get("REARCLEAN_BACKEND") or (
    f"http://{os.environ.get('REARCLEAN_HOST') or '127.0.0.1'}:"
    f"{os.environ.get('REARCLEAN_PORT') or '8000'}")


class Backend:
    def __init__(self, base_url=DEFAULT_BASE,
             timeout=int(os.environ.get("REARCLEAN_REQUEST_TIMEOUT", "1800"))):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # -- plumbing ----------------------------------------------------------

    @staticmethod
    def _clean(o):
        """Replace NaN and infinity with null before posting.

        Generated functions can return them, Flask emits them happily, and
        requests then refuses to encode them on the way back out.
        """
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: Backend._clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [Backend._clean(v) for v in o]
        return o

    def _post(self, path, payload):
        payload = self._clean(payload)
        r = requests.post(f"{self.base}{path}", json=payload, timeout=self.timeout)
        try:
            data = r.json()
        except ValueError:
            r.raise_for_status()
            raise RuntimeError(f"{path} returned a non-JSON response")
        if r.status_code >= 400 or "error" in data:
            raise RuntimeError(data.get("error", f"{path} failed with {r.status_code}"))
        return data

    @staticmethod
    def _llm(cfg):
        return {"model": cfg["model"], "endpoint": cfg["endpoint"], "token": cfg["token"]}

    # -- endpoints ---------------------------------------------------------

    def health(self):
        try:
            return requests.get(f"{self.base}/health", timeout=5).ok
        except requests.RequestException:
            return False

    def capabilities(self):
        r = requests.get(f"{self.base}/capabilities", timeout=10)
        r.raise_for_status()
        return r.json()

    def suggest(self, csv, task, cfg, error_types):
        return self._post("/suggest", {"csv": csv, "task": task,
                                       "error_types": error_types,
                                       "llm": self._llm(cfg)})

    def generate(self, csv, rules, task, cfg, tries, timeout):
        return self._post("/generate", {"csv": csv, "rules": rules, "task": task,
                                        "tries": tries, "timeout": timeout,
                                        "llm": self._llm(cfg)})

    def run(self, csv, functions, task, cfg, ordering, timeout):
        return self._post("/run", {"csv": csv, "functions": functions, "task": task,
                                   "ordering": ordering, "timeout": timeout,
                                   "llm": self._llm(cfg)})
