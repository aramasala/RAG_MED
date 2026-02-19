"""ValueAI RAG client."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValueAIRagClientConfig:
    """ValueAI RAG client configuration."""

    base_url: str
    username: str
    password: str
    rag_id: int
    model_name: str
    instructions: str = "you are helpful assistant"
    enable_metainfo: bool = True
    return_context: bool = True
    poll_interval_seconds: float = 2.0
    timeout_seconds: float = 120.0


class ValueAIRagClient:
    """Client for ValueAI RAG external API."""

    def __init__(self, config: ValueAIRagClientConfig):
        self._config = config
        self._token: str | None = None

    def _get_headers(self) -> dict[str, str]:
        if not self._token:
            self._token = self._get_token()
        return {"Authorization": f"Bearer {self._token}"}

    def _get_token(self) -> str:
        """Authenticate and return bearer token."""
        url = f"{self._config.base_url}/token"
        payload = {"username": self._config.username, "password": self._config.password}

        logger.debug(f"Getting token from {url}")

        try:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            token = data.get("authorization_token")
            if not token:
                raise ValueError(f"Token not found in response: {data}")

            logger.debug("Token received successfully")
            return token

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get token: {e}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response body: {e.response.text}")
            raise

    def create_predict(self, question: str) -> int:
        """Create a RAG prediction task and return its id."""
        url = f"{self._config.base_url}/rag/predict"
        headers = self._get_headers()
        payload = {
            "model_name": self._config.model_name,
            "request": question,
            "instructions": self._config.instructions,
            "rag_id": self._config.rag_id,
            "enable_metainfo": self._config.enable_metainfo,
            "return_context": self._config.return_context,
        }

        logger.debug(f"Creating predict task for question: {question[:50]}...")

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            predict_id = int(data["id"])
            logger.debug(f"Task created with id: {predict_id}")
            return predict_id

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create predict task: {e}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response body: {e.response.text}")
            raise

    def poll_result(self, predict_id: int) -> dict:
        """Poll a prediction until it is completed or failed."""
        url = f"{self._config.base_url}/rag/predicts/{predict_id}"
        deadline = time.monotonic() + self._config.timeout_seconds

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"ValueAI RAG predict timed out after {self._config.timeout_seconds}s"
                )

            try:
                r = requests.get(url, headers=self._get_headers(), timeout=30)

                # Token 
                if r.status_code == 401:
                    logger.debug("Token expired, refreshing...")
                    self._token = self._get_token()
                    r = requests.get(url, headers=self._get_headers(), timeout=30)

                r.raise_for_status()
                data = r.json()
                status = data.get("status")

                logger.debug(f"Task {predict_id} status: {status}")

                if status == "completed":
                    return data
                if status == "failed":
                    error_msg = data.get("error", "Unknown error")
                    result = data.get("result", {})
                    if result and "message" in result:
                        error_msg = result["message"]
                    raise RuntimeError(f"Task failed: {error_msg}")

                time.sleep(self._config.poll_interval_seconds)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error polling task: {e}")
                time.sleep(self._config.poll_interval_seconds)

    def ask(self, question: str) -> str:
        """Ask a question to ValueAI RAG and return the final response string."""
        logger.info(f"Asking ValueAI: {question[:100]}...")

        predict_id = self.create_predict(question)
        data = self.poll_result(predict_id)

        if data.get("status") != "completed":
            raise RuntimeError(f"ValueAI RAG task failed: {data}")

        result = data.get("result") or {}
        response = result.get("response")

        if not isinstance(response, str) or not response.strip():
            logger.error(f"Invalid response from ValueAI: {data}")
            raise RuntimeError("ValueAI RAG completed but response field is missing/empty")

        return response
