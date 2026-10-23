import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import stability_lab as lab


FAKE_SERVER = '''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
if "--version" in sys.argv:
    print("test server")
    sys.exit(0)
def record(value):
    with open(os.environ["TEST_SERVER_EVENTS"], "a") as f:
        f.write(json.dumps(value) + "\\n")
record({"pid": os.getpid(), "args": sys.argv[1:]})
class Handler(BaseHTTPRequestHandler):
    def reply(self, value):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(value).encode())
    def do_GET(self):
        self.reply({"status": "ok"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        record({"path": self.path, "body": body})
        if self.path == "/apply-template":
            self.reply({"prompt": "<model-template><think>"})
        elif self.path == "/tokenize":
            self.reply({"tokens": [1, 2, 3]})
        else:
            if os.environ.get("TEST_SERVER_STALL"):
                time.sleep(30)
            self.reply({"content": "reasoning\\nfinal answer", "stop_type": "eos" if body["n_predict"] == -1 else "limit",
                        "tokens_evaluated": 3, "tokens_predicted": 12, "truncated": True,
                        "generation_settings": body})
HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler).serve_forever()
'''


@unittest.skipUnless(os.name == "posix", "Executable fixture uses a POSIX shebang")
class ServerTests(unittest.TestCase):
    def config(self, root):
        binary = root / "llama-server"
        binary.write_text(f"#!{sys.executable}\n" + FAKE_SERVER)
        binary.chmod(0o755)
        model = root / "model.gguf"
        model.write_bytes(b"test fixture, not model weights")
        return {
            "backend": "llama_server", "server_binary": str(binary), "model": str(model),
            "repeats": 2, "timeout_seconds": 0,
            "env": {"TEST_SERVER_EVENTS": str(root / "events.jsonl")},
            "prompts": [{"name": "p", "text": "test", "sha256": lab.digest(b"test")}],
        }

    def test_fresh_and_warm_native_protocol_and_cleanup(self):
        for mode, expected_servers in (("fresh", 2), ("warm", 1)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.config(root)
                config["process_mode"] = mode
                report = lab.run(config, root / "run")
                self.assertTrue(report["summary"][0]["repeatable_observed"])
                self.assertEqual(report["summary"][0]["context_shifted"], 2)
                self.assertEqual(report["summary"][0]["truncated"], 0)
                events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
                servers = [e for e in events if "pid" in e]
                self.assertEqual(len(servers), expected_servers)
                requests = [e["body"] for e in events if e.get("path") == "/completion"]
                self.assertEqual(len(requests), 2)
                for request in requests:
                    self.assertEqual(request["n_predict"], -1)
                    self.assertFalse(request["cache_prompt"])
                    self.assertEqual(request["n_keep"], -1)
                for server in servers:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(server["pid"], 0)
                self.assertEqual(report["records"][0]["prompt_tokens"], [1, 2, 3])
                self.assertIn("server_sha256", report["records"][0]["runtime"])

    def test_explicit_limit_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            config.update(process_mode="warm", generate={"max_tokens": 12})
            report = lab.run(config, root / "run")
            self.assertEqual(report["summary"][0]["truncated"], 2)

    def test_chat_template_enables_reasoning_and_preserves_raw_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            config.update(process_mode="warm", reasoning=True)
            messages = [{"role": "user", "content": "Solve this."}]
            (root / "prompt.json").write_text(json.dumps(messages))
            config["prompts"] = [{"name": "p", "file": "prompt.json", "format": "chat"}]
            lab.write_json(root / "config.json", config)
            config = lab.read_config(root / "config.json")
            report = lab.run(config, root / "run")
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            template = next(e["body"] for e in events if e.get("path") == "/apply-template")
            self.assertEqual(template["messages"], messages)
            self.assertTrue(template["chat_template_kwargs"]["enable_thinking"])
            completion = next(e["body"] for e in events if e.get("path") == "/completion")
            self.assertEqual(completion["prompt"], "<model-template><think>")
            self.assertEqual(report["records"][0]["rendered_prompt"], completion["prompt"])
            self.assertEqual(report["records"][0]["text"], "reasoning\nfinal answer")
            args = events[0]["args"]
            self.assertEqual(args[args.index("--reasoning") + 1], "on")

    def test_timeout_stops_managed_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            config.update(process_mode="warm", timeout_seconds=1)
            config["env"]["TEST_SERVER_STALL"] = "1"
            report = lab.run(config, root / "run")
            self.assertEqual(report["summary"][0]["errors"], 2)
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            for event in events:
                if "pid" in event:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(event["pid"], 0)

    def test_unsupported_load_option_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(Path(tmp))
            config["load"] = {"unknown_option": True}
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                lab.server_command(lab.effective(config, {}), 12345)
