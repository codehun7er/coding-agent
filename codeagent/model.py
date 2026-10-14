import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import openai


class ModelError(Exception):
    pass


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Reply:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()

    def sanitized(self) -> "Reply":
        """Replace text that cannot be stored as UTF-8, such as lone surrogates, with a substitute character."""
        return Reply(
            _storable(self.content),
            tuple(ToolCall(_storable(c.id), _storable(c.name), _storable(c.arguments)) for c in self.tool_calls),
        )

    def message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in self.tool_calls
            ]
        return message


class Model(Protocol):
    """Anything that answers a conversation; `ref` is how a stored run finds the same model again."""

    ref: str

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> Reply: ...


class OpenAIModel:
    """Chat Completions adapter for any OpenAI-compatible endpoint, configured through OPENAI_* variables."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ref = f"openai:{name}"
        self._client: Any = None

    @classmethod
    def from_env(cls) -> "OpenAIModel":
        name = os.environ.get("OPENAI_MODEL")
        if not name:
            raise ModelError("OPENAI_MODEL is not set")
        return cls(name)

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> Reply:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY") or "unset",
            )
        try:
            response = self._client.chat.completions.create(
                model=self.name,
                messages=list(messages),
                tools=list(tools),
            )
        except openai.OpenAIError as error:
            raise ModelError(str(error)) from error
        message = response.choices[0].message
        calls = tuple(
            ToolCall(call.id or f"call_{uuid.uuid4().hex[:12]}", call.function.name, call.function.arguments or "{}")
            for call in message.tool_calls or ()
        )
        return Reply(message.content or "", calls)


class ScriptedModel:
    """Replays a fixed list of replies, choosing each by how many assistant messages the history already holds."""

    def __init__(self, steps: Sequence[Mapping[str, Any]] | None = None, *, path: Path | None = None) -> None:
        self._steps = None if steps is None else list(steps)
        self._path = path
        self.ref = f"script:{path}" if path else "script:inline"

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> Reply:
        steps = self._load()
        index = sum(1 for message in messages if message["role"] == "assistant")
        if index >= len(steps):
            raise ModelError(f"the script has no reply number {index + 1}")
        step = steps[index]
        calls = tuple(
            ToolCall(call["id"], call["name"], _encode(call.get("arguments", {})))
            for call in step.get("tool_calls", ())
        )
        return Reply(step.get("content", ""), calls)

    def _load(self) -> list[Mapping[str, Any]]:
        if self._steps is None:
            try:
                self._steps = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ModelError(f"cannot read script {self._path}: {error}") from None
        return self._steps


def load_model(ref: str) -> Model:
    kind, _, value = ref.partition(":")
    if kind == "script" and value != "inline":
        return ScriptedModel(path=Path(value))
    if kind == "openai":
        return OpenAIModel(value)
    raise ModelError(f"cannot load model {ref!r}")


def _storable(text: str) -> str:
    return text.encode(errors="replace").decode()


def _encode(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
