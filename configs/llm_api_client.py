
from __future__ import annotations
import time
from types import SimpleNamespace
import httpx
from openai import AsyncOpenAI


def get_token(base_url: str, username: str, password: str) -> str:
    url = f"{base_url.rstrip('/')}/token"
    r = httpx.post(
        url,
        json={"username": username, "password": password},
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout=30.0,
    )
    r.raise_for_status()
    data = r.json()
    token = data.get("authorization_token")
    if not token:
        raise RuntimeError("Response has no authorization_token")
    return token


def _messages_to_instructions_request(messages: list[dict]) -> tuple[str, str]:
    """Convert OpenAI-style messages to instructions (system) + request (user text)."""
    instructions_parts = []
    request_parts = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            instructions_parts.append(content)
        else:
            request_parts.append(content)
    instructions = "\n\n".join(instructions_parts) if instructions_parts else ""
    request = "\n\n".join(request_parts) if request_parts else ""
    return instructions, request


def _extract_result_text(result: str | dict | None) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "content", "response", "result", "output", "message"):
            if key in result and result[key]:
                val = result[key]
                return val if isinstance(val, str) else str(val)
        if "choices" in result and isinstance(result["choices"], list) and result["choices"]:
            first = result["choices"][0]
            if isinstance(first, dict) and "message" in first:
                msg = first["message"]
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"] or ""
    return str(result)


def predict_sync(
    base_url: str,
    token: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int = 8192,
    temperature: float | None = 0.0,
    poll_interval: float = 2.0,
    timeout: float = 120.0,
) -> str:
    """Send messages to v1/llm/predict and poll until result is ready. Returns response text."""
    instructions, request_text = _messages_to_instructions_request(messages)
    if not request_text:
        return ""

    url = f"{base_url.rstrip('/')}/llm/predict"
    payload = {
        "model_name": model_name,
        "request": request_text,
        "instructions": instructions or None,
        "temperature": temperature,
        "tokens_response_limit": max_tokens,
    }
    r = httpx.post(
        url,
        json=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    predict_id = data.get("id")
    if predict_id is None:
        raise RuntimeError("Predict response has no id")

    predict_url = f"{base_url.rstrip('/')}/llm/predicts/{predict_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r2 = httpx.get(
            predict_url,
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
            timeout=120.0,
        )
        r2.raise_for_status()
        body = r2.json()
        status = (body.get("status") or "").lower()
        result = body.get("result")
        if result is not None:
            return _extract_result_text(result)
        if status in ("failed", "error", "cancelled"):
            raise RuntimeError(f"Predict failed with status: {status}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Predict {predict_id} did not complete within {timeout}s")


async def predict_async(
    base_url: str,
    token: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int = 8192,
    temperature: float | None = 0.0,
    poll_interval: float = 2.0,
    timeout: float = 120.0,
) -> str:
    """Async: send messages to v1/llm/predict and poll until result is ready."""
    import asyncio

    instructions, request_text = _messages_to_instructions_request(messages)
    if not request_text:
        return ""

    url = f"{base_url.rstrip('/')}/llm/predict"
    payload = {
        "model_name": model_name,
        "request": request_text,
        "instructions": instructions or None,
        "temperature": temperature,
        "tokens_response_limit": max_tokens,
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            timeout=120.0,
        )
    r.raise_for_status()
    data = r.json()
    predict_id = data.get("id")
    if predict_id is None:
        raise RuntimeError("Predict response has no id")

    predict_url = f"{base_url.rstrip('/')}/llm/predicts/{predict_id}"
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            r2 = await client.get(
                predict_url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                timeout=120.0,
            )
            r2.raise_for_status()
            body = r2.json()
            status = (body.get("status") or "").lower()
            result = body.get("result")
            if result is not None:
                return _extract_result_text(result)
            if status in ("failed", "error", "cancelled"):
                raise RuntimeError(f"Predict failed with status: {status}")
            await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Predict {predict_id} did not complete within {timeout}s")


def make_async_chat_completion(content: str):
    """Build a minimal ChatCompletion-like object with choices[0].message.content and finish_reason."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


class ValueAIAsyncClient:
    """Async client that mimics OpenAI chat.completions.create for RAGAS (ValueAI v1/llm API)."""

    def __init__(
        self,
        base_url: str,
        token: str,
        model_name: str,
        max_tokens: int = 8192,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._poll_interval = poll_interval
        self._timeout = timeout
        self.chat = self
        self.completions = self

    async def create(
        self,
        model: str | None = None,
        messages: list | None = None,
        max_tokens: int | None = None,
        temperature: float | None = 0.0,
        **kwargs,
    ):
        model_name = model or self._model_name
        messages = messages or []
        max_tok = max_tokens if max_tokens is not None else self._max_tokens
        text = await predict_async(
            self._base_url,
            self._token,
            model_name,
            messages,
            max_tokens=max_tok,
            temperature=temperature,
            poll_interval=self._poll_interval,
            timeout=self._timeout,
        )
        return make_async_chat_completion(text)


class _ValueAICompletions:
    
    def __init__(self, client):
        self._client = client

    async def create(self, model=None, messages=None, max_tokens=None, temperature=None, response_model=None, **kwargs):
        model_name = model or self._client._valueai_model
        messages = messages or []
        max_tok = max_tokens or self._client._valueai_max_tokens
        text = await predict_async(
            self._client._valueai_base_url,
            self._client._valueai_token,
            model_name,
            messages,
            max_tokens=max_tok,
            temperature=temperature,
            poll_interval=self._client._valueai_poll_interval,
            timeout=self._client._valueai_timeout,
        )
        return make_async_chat_completion(text)


class _ValueAIChat:
    def __init__(self, client):
        self._client = client
        self.completions = _ValueAICompletions(client)


class ValueAIAsyncOpenAI(AsyncOpenAI):
    
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        model_name: str,
        max_tokens: int = 8192,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
        **kwargs,
    ):
        super().__init__(
            api_key=token,
            base_url=base_url,
            default_headers={"Authorization": f"Bearer {token}"},
            **kwargs,
        )
        self._valueai_base_url = base_url.rstrip("/")
        self._valueai_token = token
        self._valueai_model = model_name
        self._valueai_max_tokens = max_tokens
        self._valueai_poll_interval = poll_interval
        self._valueai_timeout = timeout

    @property
    def chat(self):
        return _ValueAIChat(self)
