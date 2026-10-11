# claudecodestreaming

Stream text from the [Claude Code](https://claude.com/claude-code) CLI. This library is a text-completion alternative to [agentmake](https://github.com/eliranwong/agentmake): you pass chat messages, and it yields response text as the CLI produces it.

Authentication is the subscription session from `claude auth login`, or a subscription token from `claude setup-token` exported as `ANTHROPIC_AUTH_TOKEN`. The child process does not receive `ANTHROPIC_API_KEY`. No API key is read or sent.

## Install

```bash
pip install claudecodestreaming
```

Requires the `claude` executable on `PATH`, at `~/.claude/local/claude`, or at `~/.local/bin/claude`, and a completed `claude auth login`.

## Stream

```python
import asyncio
from claudecodestreaming import stream_text

async def main():
    messages = [
        {"role": "system", "content": "You are a concise Bible study assistant."},
        {"role": "user", "content": "What is John 3:16 about?"},
        {"role": "assistant", "content": "It states God's love for the world."},
    ]
    async for chunk in stream_text(messages, follow_up="Quote it briefly."):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

`messages` use `role` and `content`, the same shape agentmake accepts. `follow_up` is the newest user turn. An explicit `system` argument is sent with `--system-prompt`. System messages already in `messages` stay in the user prompt, so a generic identity does not outrank that task. Each call is a new headless process: the library does not resume a previous Claude Code session.

The process is text-only. Every built-in tool is removed (`--tools ""`), MCP servers are not loaded (`--strict-mcp-config`), session persistence is off, and it runs in a temporary working directory; with no tools there is exactly one turn. The prompt travels on stdin, so it never appears in `ps` output. Cancel an in-flight stream by setting the `asyncio.Event` passed as `cancel_event`.

`model` defaults to the CLI's configured model, so the library follows whatever `claude` is signed in and set up to run. Pass `model="sonnet"` or a full model name to pin one.

`collect_text` returns the whole reply:

```python
from claudecodestreaming import collect_text

text = await collect_text(follow_up="Reply with exactly: pong")
```