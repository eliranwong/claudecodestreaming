"""Headless Claude Code text streaming.

Each call starts one ``claude`` process and yields ``text_delta`` chunks from
``--output-format stream-json`` with ``--include-partial-messages``. The
process uses the signed-in subscription session. ``ANTHROPIC_API_KEY`` is
removed from the child environment so an exported API key is not the
credential; ``ANTHROPIC_AUTH_TOKEN`` is kept, because a subscription token
from ``claude setup-token`` may live there.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

# None means the CLI's configured default model, so the library works with any
# sign-in. Pass model="sonnet" or a full model name to pin one.
DEFAULT_MODEL: str | None = None
STREAM_LIMIT = 8 * 1024 * 1024
# ANTHROPIC_API_KEY outranks the login session and would bill an API key
# instead of the subscription, so it is removed on purpose.
# ANTHROPIC_AUTH_TOKEN is kept: a subscription token from ``claude
# setup-token`` is exported there, and removing it would log the child out.
_API_KEY_ENV = ("ANTHROPIC_API_KEY",)
# The system prompt owns the task. The user message is the input to that task.
_TEXT_ONLY_RULE = (
    "Do not call tools, read or edit files, browse the web, or run commands. "
    "Obey the system prompt. The user message is the input to that task. "
    "When the system prompt asks for a specification or a formatted block instead of a direct answer, produce only that specification."
)


class ClaudeCodeError(RuntimeError):
    """The Claude Code CLI is missing, or a turn failed before any text."""


def find_claude_bin() -> str:
    """Return the ``claude`` executable from ``PATH`` or a known install path."""
    found = shutil.which("claude")
    if found and os.path.isfile(found) and os.access(found, os.X_OK):
        return found
    for location in ("~/.claude/local/claude", "~/.local/bin/claude"):
        fallback = os.path.expanduser(location)
        if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
            return fallback
    raise ClaudeCodeError(
        "The claude CLI was not found. Install Claude Code, then sign in with: claude auth login"
    )


def subscription_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a child process that keeps the ``claude`` login session."""
    env = dict(os.environ if base is None else base)
    for name in _API_KEY_ENV:
        env.pop(name, None)
    env["DISABLE_AUTOUPDATER"] = "1"
    return env


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if not isinstance(text, str):
                    text = item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def format_prompt(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
) -> tuple[str, str]:
    """Split chat messages into a user prompt and a system prompt.

    A lone user turn is left unlabeled so the model sees that text exactly.
    Earlier turns are labeled when history is present. The explicit ``system``
    argument is the system prompt. System messages already in ``messages`` stay
    in the user prompt, so a generic identity does not outrank the task.
    """
    system_parts: list[str] = []
    if isinstance(system, str) and system.strip():
        system_parts.append(system.strip())
    standing: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages or ():
        role = str(message.get("role") or "user").strip().lower()
        content = _content_to_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            standing.append(content)
        elif role == "assistant":
            turns.append(("Assistant", content))
        else:
            turns.append(("User", content))
    if isinstance(follow_up, str) and follow_up.strip():
        turns.append(("User", follow_up.strip()))
    system_parts.append(_TEXT_ONLY_RULE)
    if not turns and not standing:
        raise ClaudeCodeError("No prompt to send.")
    if len(turns) == 1 and turns[0][0] == "User":
        prompt = turns[0][1]
    else:
        prompt = "\n\n".join(f"{role}:\n{content}" for role, content in turns)
    if standing:
        context = "\n\n".join(standing)
        prompt = f"{context}\n\n{prompt}" if prompt else context
    return prompt, "\n\n".join(system_parts)


def build_command(
    claude_bin: str,
    system_prompt: str,
    model: str | None = None,
) -> list[str]:
    """Argv for one text-only headless turn. No API key and no tool approval.

    ``--tools ""`` removes every built-in tool, so the turn cannot branch and
    no permission can be asked. ``--strict-mcp-config`` with no ``--mcp-config``
    loads no MCP servers.
    """
    command = [
        claude_bin,
        "--print",
        "--verbose",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--permission-mode", "bypassPermissions",
        "--system-prompt", system_prompt,
    ]
    chosen_model = model.strip() if isinstance(model, str) and model.strip() else DEFAULT_MODEL
    if chosen_model:
        command.extend(("--model", chosen_model))
    return command


def parse_stream_line(line: str) -> dict[str, Any] | None:
    """Parse one stdout line from ``--output-format stream-json``."""
    text = line.strip()
    if not text:
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(event, dict):
        return event
    return None


def text_from_event(event: Mapping[str, Any]) -> str | None:
    """Return a streamed text chunk. Other event types contribute no answer text."""
    if event.get("type") != "stream_event":
        return None
    inner = event.get("event")
    if not isinstance(inner, dict) or inner.get("type") != "content_block_delta":
        return None
    delta = inner.get("delta")
    if isinstance(delta, dict) and delta.get("type") == "text_delta":
        text = delta.get("text")
        if isinstance(text, str) and text:
            return text
    return None


def message_text_from_event(event: Mapping[str, Any]) -> str | None:
    """Return text from a whole ``assistant`` message event.

    This is the fallback for CLI versions that do not stream partial
    messages. API error messages are not answer text and yield ``None``.
    """
    if event.get("type") != "assistant" or event.get("is_api_error_message"):
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts) or None


def _completion_workspace() -> str:
    """Stable directory outside the caller's project, with no git root of its own."""
    path = os.path.join(tempfile.gettempdir(), "claudecodestreaming-workspace")
    os.makedirs(path, exist_ok=True)
    return path


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        return


async def _next_line(stream: asyncio.StreamReader, cancel_event: asyncio.Event | None) -> bytes | None:
    """Read one stdout line. ``None`` means the caller cancelled."""
    read_task = asyncio.create_task(stream.readline())
    if cancel_event is None:
        return await read_task
    if cancel_event.is_set():
        await _cancel_task(read_task)
        return None
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {read_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        await _cancel_task(read_task)
        await _cancel_task(cancel_task)
        raise
    if read_task in done and not read_task.cancelled():
        await _cancel_task(cancel_task)
        return read_task.result()
    await _cancel_task(read_task)
    await _cancel_task(cancel_task)
    return None


async def _read_stderr(stream: asyncio.StreamReader | None) -> str:
    if stream is None:
        return ""
    chunks: list[str] = []
    size = 0
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            chunks.append(text)
            size += len(text)
            if size > 16000:
                chunks = chunks[-20:]
                size = sum(len(item) for item in chunks)
    except asyncio.CancelledError:
        return "".join(chunks)[-8000:]
    return "".join(chunks)[-8000:]


async def _stop_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None or process.returncode is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            except Exception:
                process.kill()
        try:
            await process.wait()
        except Exception:
            return


async def stream_text(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
    model: str | None = None,
    claude_bin: str | None = None,
    cancel_event: asyncio.Event | None = None,
):
    """Yield text chunks from one signed-in Claude Code turn.

    Closing the generator, or setting ``cancel_event``, stops the CLI process.
    ``model`` defaults to :data:`DEFAULT_MODEL` (the CLI's configured model).
    The working directory is a temporary folder, and this call does not resume
    an earlier session.
    """
    prompt, system_prompt = format_prompt(messages, follow_up=follow_up, system=system)
    executable = claude_bin or find_claude_bin()
    work_dir = _completion_workspace()
    process: asyncio.subprocess.Process | None = None
    stderr_task: asyncio.Task | None = None
    produced_text = False
    try:
        command = build_command(executable, system_prompt, model=model)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=subscription_env(),
            start_new_session=True,
            limit=STREAM_LIMIT,
        )
        if process.stdin is None or process.stdout is None:
            raise ClaudeCodeError("claude did not provide usable streams")
        # The prompt travels on stdin: no argv length limit, nothing in `ps`.
        try:
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        process.stdin.close()
        stderr_task = asyncio.create_task(_read_stderr(process.stderr))
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return
            line = await _next_line(process.stdout, cancel_event)
            if line is None:
                return
            if line == b"":
                break
            event = parse_stream_line(line.decode("utf-8", errors="replace"))
            if not event:
                continue
            if event.get("type") == "result":
                if cancel_event is not None and cancel_event.is_set():
                    return
                if event.get("is_error"):
                    message = event.get("result") or "Claude Code reported an error."
                    raise ClaudeCodeError(str(message))
                # The result repeats the whole reply once the deltas are done.
                if not produced_text:
                    text = event.get("result")
                    if isinstance(text, str) and text:
                        produced_text = True
                        yield text
                continue
            chunk = text_from_event(event)
            if chunk:
                produced_text = True
                yield chunk
                continue
            # Without partial messages the whole assistant message is the text.
            if not produced_text:
                chunk = message_text_from_event(event)
                if chunk:
                    produced_text = True
                    yield chunk
        returncode = await process.wait()
        stderr_text = ""
        if stderr_task is not None:
            stderr_text = (await stderr_task).strip()
            stderr_task = None
        if returncode not in (0, None) and not produced_text:
            detail = stderr_text or f"claude exited with code {returncode}"
            raise ClaudeCodeError(detail)
    finally:
        await _stop_process(process)
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass


async def collect_text(
    messages: Sequence[Mapping[str, Any]] | None = None,
    follow_up: str | None = None,
    system: str | None = None,
    model: str | None = None,
    claude_bin: str | None = None,
    cancel_event: asyncio.Event | None = None,
) -> str:
    """Run :func:`stream_text` and return the joined text."""
    chunks: list[str] = []
    async for chunk in stream_text(
        messages,
        follow_up=follow_up,
        system=system,
        model=model,
        claude_bin=claude_bin,
        cancel_event=cancel_event,
    ):
        chunks.append(chunk)
    return "".join(chunks)