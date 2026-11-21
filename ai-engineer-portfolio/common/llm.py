"""Tiny LLM wrapper. With ANTHROPIC_API_KEY set it calls a real model;
otherwise it returns a stub so every pipeline still runs offline.
Swap the body for OpenAI / Bedrock / Ollama - nothing else in the repo changes."""
import json
import os
import re


def llm(prompt: str, system: str = "You are a precise assistant.") -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        r = client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text
    return "[stub-llm] " + prompt[-300:]


def llm_json(prompt: str, default=None) -> dict:
    out = llm(prompt + "\nReturn ONLY valid JSON.")
    m = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(m.group(0)) if m else (default or {})
    except json.JSONDecodeError:
        return default or {}
