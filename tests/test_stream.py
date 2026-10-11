import unittest
from unittest import mock

from claudecodestreaming.stream import (
    DEFAULT_MODEL,
    ClaudeCodeError,
    build_command,
    find_claude_bin,
    format_prompt,
    message_text_from_event,
    parse_stream_line,
    subscription_env,
    text_from_event,
)


class FormatPromptTests(unittest.TestCase):
    def test_single_user_turn_is_unlabeled(self):
        prompt, system = format_prompt(
            [{"role": "system", "content": "You are BibleMate AI."}],
            follow_up="List two tools.",
        )
        self.assertIn("You are BibleMate AI.", prompt)
        self.assertTrue(prompt.endswith("List two tools."))
        self.assertNotIn("You are BibleMate AI.", system)
        self.assertNotIn("List two tools.", system)
        self.assertIn("Obey the system prompt.", system)
        self.assertNotIn("User:", prompt)

    def test_history_keeps_the_latest_request_intact(self):
        prompt, system = format_prompt(
            [
                {"role": "system", "content": "Role."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
            ],
            follow_up="Second question",
            system="Extra role",
        )
        self.assertIn("Role.", prompt)
        self.assertIn("User:\nFirst question", prompt)
        self.assertIn("Assistant:\nFirst answer", prompt)
        self.assertTrue(prompt.endswith("User:\nSecond question"))
        self.assertIn("Extra role", system)
        self.assertNotIn("Role.", system)
        self.assertNotIn("Second question", system)

    def test_content_blocks_join_text(self):
        prompt, _system = format_prompt(
            [{"role": "user", "content": [{"type": "text", "text": "Hello"}, {"text": "there"}]}],
        )
        self.assertEqual(prompt, "Hello\nthere")

    def test_empty_prompt_raises(self):
        with self.assertRaises(ClaudeCodeError):
            format_prompt([])


class EventTests(unittest.TestCase):
    def test_text_delta_chunk(self):
        event = parse_stream_line(
            '{"type":"stream_event","event":{"type":"content_block_delta",'
            '"index":0,"delta":{"type":"text_delta","text":"pong"}}}'
        )
        self.assertEqual(text_from_event(event), "pong")

    def test_thinking_delta_adds_no_answer_text(self):
        event = parse_stream_line(
            '{"type":"stream_event","event":{"type":"content_block_delta",'
            '"index":0,"delta":{"type":"thinking_delta","thinking":"hm"}}}'
        )
        self.assertIsNone(text_from_event(event))

    def test_tool_event_adds_no_answer_text(self):
        event = parse_stream_line(
            '{"type":"assistant","message":{"content":'
            '[{"type":"tool_use","id":"t1","name":"Bash","input":{}}]}}'
        )
        self.assertIsNone(message_text_from_event(event))

    def test_assistant_error_message_adds_no_answer_text(self):
        event = parse_stream_line(
            '{"type":"assistant","is_api_error_message":true,"message":{"content":'
            '[{"type":"text","text":"There was an issue with the selected model."}]}}'
        )
        self.assertIsNone(message_text_from_event(event))

    def test_assistant_text_is_the_fallback(self):
        event = parse_stream_line(
            '{"type":"assistant","message":{"content":'
            '[{"type":"text","text":"pong"}]}}'
        )
        self.assertEqual(message_text_from_event(event), "pong")

    def test_non_json_is_ignored(self):
        self.assertIsNone(parse_stream_line("plain log line"))


class CommandTests(unittest.TestCase):
    def test_command_is_text_only_and_has_no_api_key(self):
        command = build_command("/usr/bin/claude", "Be brief.")
        joined = " ".join(command)
        self.assertNotIn("ANTHROPIC_API_KEY", joined)
        self.assertIn("--print", command)
        self.assertIn("--verbose", command)
        self.assertIn("--output-format", command)
        self.assertIn("stream-json", command)
        self.assertIn("--include-partial-messages", command)
        self.assertIn("--tools", command)
        self.assertIn("", command)
        self.assertIn("--strict-mcp-config", command)
        self.assertIn("--no-session-persistence", command)
        self.assertIn("--permission-mode", command)
        self.assertIn("bypassPermissions", command)
        self.assertIn("--system-prompt", command)
        self.assertIn("Be brief.", command)
        # The CLI's configured model is the default, so no model is pinned.
        self.assertNotIn("--model", command)

    def test_explicit_model_is_pinned(self):
        command = build_command("/usr/bin/claude", "Be brief.", model="sonnet")
        self.assertIn("--model", command)
        self.assertIn("sonnet", command)

    def test_default_model_is_none(self):
        self.assertIsNone(DEFAULT_MODEL)

    def test_subscription_env_drops_api_key_only(self):
        env = subscription_env(
            {"ANTHROPIC_API_KEY": "sk-ant-secret", "ANTHROPIC_AUTH_TOKEN": "oauth-token", "PATH": "/usr/bin", "HOME": "/Users/me"}
        )
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        # A subscription token from `claude setup-token` is the credential.
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "oauth-token")
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["DISABLE_AUTOUPDATER"], "1")

    def test_find_claude_bin_uses_path(self):
        with mock.patch("claudecodestreaming.stream.shutil.which", return_value="/opt/claude"):
            with mock.patch("claudecodestreaming.stream.os.path.isfile", return_value=True):
                with mock.patch("claudecodestreaming.stream.os.access", return_value=True):
                    self.assertEqual(find_claude_bin(), "/opt/claude")

    def test_find_claude_bin_uses_known_install_fallback(self):
        with mock.patch("claudecodestreaming.stream.shutil.which", return_value=None):
            with mock.patch("claudecodestreaming.stream.os.path.expanduser", side_effect=lambda p: p):
                with mock.patch("claudecodestreaming.stream.os.path.isfile", return_value=True):
                    with mock.patch("claudecodestreaming.stream.os.access", return_value=True):
                        self.assertEqual(find_claude_bin(), "~/.claude/local/claude")

    def test_find_claude_bin_errors_when_missing(self):
        with mock.patch("claudecodestreaming.stream.shutil.which", return_value=None):
            with mock.patch("claudecodestreaming.stream.os.path.expanduser", side_effect=lambda p: p):
                with mock.patch("claudecodestreaming.stream.os.path.isfile", return_value=False):
                    with self.assertRaises(ClaudeCodeError):
                        find_claude_bin()


if __name__ == "__main__":
    unittest.main()