from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx

# azure-identity is only needed when you enable Azure Foundry mode.
from azure.identity import DefaultAzureCredential


def _get_token() -> str:
    """
    Acquire an Entra ID access token for Azure AI Foundry Agent Service.

    Works with:
      - Managed Identity (recommended on Azure App Service)
      - EnvironmentCredential (AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET)
      - Azure CLI credential (local dev)
    """
    cred = DefaultAzureCredential()
    token = cred.get_token("https://ai.azure.com/.default")
    return token.token


def _extract_text_from_message(msg: Dict[str, Any]) -> str:
    """
    Robustly extract text from Agent Service message payloads.

    Supports common shapes:
      - {"content":[{"type":"text","text":{"value":"..."}}]}
      - {"content":[{"type":"output_text","text":"..."}]}
      - {"content":"..."}
    """
    content = msg.get("content")
    if content is None:
        return ""

    # simple string
    if isinstance(content, str):
        return content.strip()

    # list of parts
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("text", "output_text"):
                txt = ""
                t = part.get("text")
                if isinstance(t, dict):
                    txt = t.get("value") or t.get("text") or ""
                elif isinstance(t, str):
                    txt = t
                if txt:
                    chunks.append(str(txt).strip())
            # fallback keys
            if not chunks:
                maybe = part.get("value") or part.get("content") or ""
                if maybe:
                    chunks.append(str(maybe).strip())
        return "\n\n".join([c for c in chunks if c]).strip()

    # fallback
    return str(content).strip()


def ask_foundry_agent(
    user_text: str,
    *,
    project_endpoint: Optional[str] = None,
    assistant_id: Optional[str] = None,
    api_version: str = "v1",
    poll_s: float = 0.6,
    timeout_s: float = 90.0,
) -> str:
    """
    Send a single-turn request to an Azure AI Foundry Agent.

    You MUST set (as env vars or pass explicitly):
      - AI_FOUNDRY_PROJECT_ENDPOINT  e.g. https://<resource>.services.ai.azure.com/api/projects/<project-name>
      - AI_FOUNDRY_ASSISTANT_ID

    Returns the latest assistant text.
    """
    endpoint = (project_endpoint or os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT", "")).strip().rstrip("/")
    a_id = (assistant_id or os.getenv("AI_FOUNDRY_ASSISTANT_ID", "")).strip()
    if not endpoint or not a_id:
        raise RuntimeError("Azure Foundry is not configured: set AI_FOUNDRY_PROJECT_ENDPOINT and AI_FOUNDRY_ASSISTANT_ID")

    token = _get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    with httpx.Client(timeout=30.0) as client:
        # 1) Create thread + run
        r = client.post(
            f"{endpoint}/threads/runs",
            params={"api-version": api_version},
            headers=headers,
            json={
                "assistant_id": a_id,
                "thread": {"messages": [{"role": "user", "content": user_text}]},
            },
        )
        r.raise_for_status()
        run_obj = r.json()

        thread_id = run_obj.get("thread_id") or run_obj.get("thread", {}).get("id")
        run_id = run_obj.get("id")
        if not thread_id or not run_id:
            raise RuntimeError(f"Unexpected run response (missing ids): {run_obj}")

        # 2) Poll run status
        deadline = time.time() + timeout_s
        status = None
        while time.time() < deadline:
            rr = client.get(
                f"{endpoint}/threads/{thread_id}/runs/{run_id}",
                params={"api-version": api_version},
                headers=headers,
            )
            rr.raise_for_status()
            status_obj = rr.json()
            status = status_obj.get("status")
            if status == "completed":
                break
            if status in ("failed", "cancelled", "expired"):
                raise RuntimeError(f"Run ended with status={status}: {status_obj}")
            time.sleep(poll_s)

        if status != "completed":
            raise RuntimeError(f"Run timed out after {timeout_s}s (last status={status})")

        # 3) Read messages and return newest assistant
        mr = client.get(
            f"{endpoint}/threads/{thread_id}/messages",
            params={"api-version": api_version},
            headers=headers,
        )
        mr.raise_for_status()
        payload = mr.json()
        data = payload.get("data") or payload.get("messages") or []
        # usually newest-first, but be safe
        assistant_msgs = [m for m in data if isinstance(m, dict) and m.get("role") == "assistant"]
        if not assistant_msgs and data:
            # sometimes role might be nested / different; fallback to scanning all
            assistant_msgs = [m for m in data if isinstance(m, dict) and "assistant" in str(m.get("role", "")).lower()]

        if not assistant_msgs:
            return ""

        text = _extract_text_from_message(assistant_msgs[0])
        return text or ""
