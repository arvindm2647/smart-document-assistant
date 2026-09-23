"""
llm_client.py
--------------
Thin wrapper around the Groq API so the rest of the app doesn't need to
know which LLM provider is behind it. Groq offers a free API key with
generous rate limits, which is why it's used here.

Swapping to a different provider later (Anthropic, OpenAI, Gemini) would
only require changing this one file -- nothing in qa.py or app.py needs
to know which provider is behind `generate()`.
"""

import os
from groq import Groq

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to a .env file or your environment. "
                "Get a free key at https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.2,  # low temperature keeps answers grounded, not creative
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()
