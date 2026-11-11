"""LLM access.

A single place where the model is called, so that every part of the backend
talks to it the same way and nothing else needs the credentials.

Reached over the OpenAI protocol, so hosted and locally served models are
interchangeable.
"""

import os
import re

DEFAULT_MAX_TOKENS = int(os.environ.get("REARCLEAN_MAX_TOKENS", "4096"))
DEFAULT_TIMEOUT = int(os.environ.get("REARCLEAN_LLM_TIMEOUT", "60"))
DEFAULT_RETRIES = int(os.environ.get("REARCLEAN_LLM_RETRIES", "2"))


class LLM:
    """Callable wrapper around a chat model.

    Call it as llm(human_prompt, system_prompt). Responses are cached on the
    exact prompt and temperature, so a repeated identical call costs nothing;
    temperature is part of the key, so a retry that only raises temperature is
    not served the deterministic answer.
    """

    def __init__(self, model, endpoint, token, max_tokens=DEFAULT_MAX_TOKENS,
                 timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_RETRIES):
        self.model = model
        self.endpoint = endpoint
        self.token = token
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.calls = []          # transcript, so a run can be inspected afterwards
        self._cache = {}

    def __call__(self, human, system, temperature=0.0):
        key = (system, human, temperature)
        if key in self._cache:
            return self._cache[key]

        # imported here rather than at module load, so the service still starts
        # and serves the endpoints that need no model if langchain is absent
        try:
            from langchain_community.chat_models import ChatOpenAI
        except ImportError as e:
            raise RuntimeError(
                "langchain-community is not installed, so no model can be "
                "called. Install it with: pip install langchain-community") from e

        chat = ChatOpenAI(
            model_name=self.model,
            openai_api_base=self.endpoint,
            openai_api_key=self.token,
            temperature=temperature,
            # a cap, not a charge: only tokens actually generated cost anything,
            # and rule generation can run long
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            # rate limits and transient 5xx are worth retrying on a hosted
            # endpoint; lower this if the endpoint tends to hang
            max_retries=self.max_retries,
        )

        # tuple form, so no message classes need importing: langchain has moved
        # them between packages and this works across versions
        messages = [("system", system), ("human", human)]
        response = chat.invoke(messages).content

        if response and looks_truncated(response):
            print(f"\tWARNING: response may be truncated at {self.max_tokens} tokens "
                  f"({len(response)} chars, no clean ending)")

        self.calls.append({"system": system, "human": human,
                           "response": response, "temperature": temperature})
        self._cache[key] = response
        return response


def looks_truncated(response):
    """Cheap check for a response cut off mid-sentence by the token cap."""
    tail = response.rstrip()
    if not tail:
        return False
    if tail.count("```") % 2 == 1:      # opened a code block and never closed it
        return True
    return bool(re.search(r"[,(\[{]$", tail))