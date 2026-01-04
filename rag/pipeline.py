from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from .schema import RegulAIteAnswer, DEFAULT_EMPTY
from .foundry_client import ask_foundry_agent


def _strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^```(?:json|yaml|yml|text)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json(text: str) -> Dict[str, Any]:
    """Best-effort: if the agent returns JSON, parse it; otherwise return {}."""
    if not text:
        return {}
    text = _strip_code_fences(text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        # tolerate trailing commas
        raw2 = re.sub(r",\s*}", "}", raw)
        raw2 = re.sub(r",\s*]", "]", raw2)
        try:
            return json.loads(raw2)
        except Exception:
            return {}


def _history_to_brief(history: Optional[List[Dict[str, str]]], max_turns: int = 10) -> str:
    if not history:
        return ""
    # keep only latest messages; compress to text
    lines: List[str] = []
    for m in history[-max_turns:]:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _env_first(*names: str) -> str:
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v:
            return v
    return ""


def ask(
    query: str,
    *,
    user_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    k_hint: int = 12,
    evidence_mode: bool = True,
    mode_hint: str | None = "long",
    web_enabled: Union[bool, str] = False,
    vec_id: Optional[str] = None,
    model: Optional[str] = None,
) -> RegulAIteAnswer:
    """
    AGENT-ONLY PIPELINE

    This app is intentionally "UI only":
    - No third-party model API keys
    - No internet search
    - No local retrieval logic

    It forwards the user query (plus a short conversation brief) to the published
    Azure AI Foundry Agent and renders whatever the Agent returns.
    """

    # Support both naming styles so you can keep your existing App Service settings.
    project_endpoint = _env_first(
        "AI_FOUNDRY_PROJECT_ENDPOINT",
        "AZURE_EXISTING_AIPROJECT_ENDPOINT",
        "AI_FOUNDRY_PROJECT_ENDPOINT",
    )
    assistant_id = _env_first(
        "AI_FOUNDRY_ASSISTANT_ID",
        "AZURE_EXISTING_AGENT_ID",
        "AZURE_EXISTING_AGENT_ID",
    )

    if not project_endpoint or not assistant_id:
        missing = []
        if not project_endpoint:
            missing.append("AI_FOUNDRY_PROJECT_ENDPOINT (or AZURE_EXISTING_AIPROJECT_ENDPOINT)")
        if not assistant_id:
            missing.append("AI_FOUNDRY_ASSISTANT_ID (or AZURE_EXISTING_AGENT_ID)")
        return RegulAIteAnswer(
            raw_markdown=(
                "### Configuration error\n"
                "This website is configured to use an **Azure Foundry Agent only**, but required environment variables are missing.\n\n"
                f"Missing: {', '.join(missing)}"
            )
        )

    convo_brief = _history_to_brief(history)
    user_text = query
    if convo_brief:
        user_text = f"Conversation brief (latest first):\n{convo_brief}\n\nUser query:\n{query}"

    raw = ask_foundry_agent(
        user_text,
        project_endpoint=project_endpoint,
        assistant_id=assistant_id,
        api_version=os.getenv("AI_FOUNDRY_API_VERSION", "v1"),
    )

    if not raw:
        return DEFAULT_EMPTY

    data = _parse_json(raw)
    if data:
        try:
            return RegulAIteAnswer(**data)
        except ValidationError:
            # If the agent returned some JSON-ish structure but it doesn't validate,
            # fall back to showing it as plain text.
            return RegulAIteAnswer(raw_markdown=raw.strip())

    return RegulAIteAnswer(raw_markdown=raw.strip())
