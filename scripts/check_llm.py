"""Step 3 of the spec: User -> LLM -> Response.

Sends one tiny request through the configured provider so you can confirm
your API key and model name work before running a full investigation.

    docker compose exec api python -m scripts.check_llm
    (or locally)  python -m scripts.check_llm
"""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from app.llm.base import LLMError, LLMPing
from app.llm.factory import get_llm


async def main() -> int:
    settings = get_settings()
    print(f"Provider: {settings.LLM_PROVIDER}   Model: {settings.llm_model_name or '(not set)'}")
    try:
        llm = get_llm()
        answer = await llm.structured(
            LLMPing,
            system="You are a connectivity test. Answer briefly.",
            user="Reply with the single word: pong",
            purpose="connectivity_check",
        )
    except LLMError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"OK - the model answered: {answer.reply!r}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
