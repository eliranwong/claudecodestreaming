"""Stream text from the signed-in Claude Code CLI.

The CLI session created by ``claude auth login`` is the credential. This
package does not read or send an API key.
"""

from claudecodestreaming.stream import (
    DEFAULT_MODEL,
    ClaudeCodeError,
    collect_text,
    find_claude_bin,
    format_prompt,
    message_text_from_event,
    stream_text,
    subscription_env,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MODEL",
    "ClaudeCodeError",
    "collect_text",
    "find_claude_bin",
    "format_prompt",
    "message_text_from_event",
    "stream_text",
    "subscription_env",
]