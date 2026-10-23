"""Local output stability experiments using llama-server or llama-cpp-python."""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fingerprint():
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                result["cpu_model"] = line.partition(":")[2].strip()
                break
    return result


def resolved_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def read_config(path: Path, model_override=None, repeats_override=None):
    config = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    unknown = set(config) - {"model", "repeats", "timeout_seconds", "process_mode", "load", "generate", "prompts", "variants", "python", "env", "backend", "server_binary", "reasoning"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if repeats_override is not None:
        config["repeats"] = repeats_override
    if not isinstance(config.get("repeats", 3), int) or config.get("repeats", 3) < 2:
        raise ValueError("repeats must be an integer >= 2 to measure stability")
    if config.get("process_mode", "fresh") not in ("fresh", "warm"):
        raise ValueError("process_mode must be fresh or warm")
    if config.get("timeout_seconds", 1800) < 0:
        raise ValueError("timeout_seconds must be nonnegative (0 disables the timeout)")
    names = {"baseline"}
    for variant in config.get("variants", []):
        if not variant.get("name") or variant["name"] in names:
            raise ValueError("variant names must be unique; baseline is reserved")
        names.add(variant["name"])
        unknown = set(variant) - {"name", "load", "generate", "model", "python", "env"}
        if unknown:
            raise ValueError(f"Unknown variant keys: {sorted(unknown)}")
    prompts = config.get("prompts", [])
    if not prompts or len({p["name"] for p in prompts}) != len(prompts):
        raise ValueError("provide at least one prompt, with unique names")
    for prompt in prompts:
        # Decode bytes directly: preserve CRLF rather than silently normalizing it.
        prompt["text"] = resolved_path(base, prompt["file"]).read_bytes().decode("utf-8")
        prompt["sha256"] = digest(prompt["text"].encode("utf-8"))
        if prompt.get("format", "raw") == "chat":
            prompt["messages"] = json.loads(prompt["text"])
            if not isinstance(prompt["messages"], list) or not prompt["messages"]:
                raise ValueError("Chat prompt files must contain a nonempty JSON array of messages")
            if config.get("backend") != "llama_server":
                raise ValueError("Chat prompts require the llama_server backend")
        elif prompt.get("format", "raw") != "raw":
            raise ValueError("Prompt format must be raw or chat")
    if model_override:
        config["model"] = str(Path(model_override).expanduser().resolve())
    if config.get("backend", "llama_cpp") != "demo":
        config["model"] = str(resolved_path(base, config["model"]))
        for variant in config.get("variants", []):
            if "model" in variant:
                variant["model"] = str(resolved_path(base, variant["model"]))
        for model in {config["model"]} | {v["model"] for v in config.get("variants", []) if "model" in v}:
            if not Path(model).is_file():
                raise ValueError(f"Model does not exist: {model}. Use --model /path/to/model.gguf")
    return config


def effective(config, variant):
    load = {"n_ctx": 4096, "n_threads": 1, "n_threads_batch": 1, "n_gpu_layers": 0,
            **config.get("load", {}), **variant.get("load", {})}
    generate = {"seed": 42, "temperature": 0.0, "max_tokens": -1 if config.get("backend") == "llama_server" else 512,
                **config.get("generate", {}), **variant.get("generate", {})}
    # Streaming and echo would change the worker's result schema/comparison semantics.
    if generate.get("stream") or generate.get("echo"):
        raise ValueError("stream and echo must be false")
    if "prompt" in generate or "model_path" in load:
        raise ValueError("Use prompts/model fields, not generate.prompt or load.model_path")
    return {
        "backend": config.get("backend", "llama_cpp"),
        "model": variant.get("model", config.get("model")),
        "load": load,
        "generate": generate,
        "python": variant.get("python", config.get("python", sys.executable)),
        "env": {**config.get("env", {}), **variant.get("env", {})},
        "server_binary": config.get("server_binary", "llama-server"),
        "reasoning": config.get("reasoning"),
    }


def server_command(req, port):
    binary = shutil.which(req["server_binary"])
    if not binary:
        raise ValueError(f"llama-server binary not found: {req['server_binary']}")
    command = [binary, "--model", req["model"], "--host", "127.0.0.1", "--port", str(port),
               "--parallel", "1", "--no-cache-prompt", "--context-shift", "--n-predict", "-1"]
    if req.get("reasoning") is not None:
        if not isinstance(req["reasoning"], bool):
            raise ValueError("reasoning must be true or false")
        command += ["--reasoning", "on" if req["reasoning"] else "off", "--reasoning-budget", "-1"]
    flags = {"n_ctx": "--ctx-size", "n_threads": "--threads", "n_threads_batch": "--threads-batch",
             "n_gpu_layers": "--gpu-layers", "n_batch": "--batch-size", "n_ubatch": "--ubatch-size"}
    unknown = set(req["load"]) - set(flags) - {"use_mmap", "flash_attn"}
    if unknown:
        raise ValueError(f"Unsupported llama_server load settings: {sorted(unknown)}")
    for key, value in req["load"].items():
        if key in flags:
            command += [flags[key], "all" if key == "n_gpu_layers" and value == -1 else str(value)]
        elif key == "use_mmap":
            command += ["--load-mode", "mmap" if value else "none"]
        elif key == "flash_attn":
            command += ["--flash-attn", "on" if value else "off"]
    return command


def server_generate(req, result):
    # A separate local server for each worker preserves fresh/warm experiment semantics.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = server_command(req, port)
    generate = dict(req["generate"])
    supported = {"max_tokens", "seed", "temperature", "top_k", "top_p", "min_p", "repeat_penalty",
                 "repeat_last_n", "typical_p", "stop", "stream", "echo"}
    if set(generate) - supported:
        raise ValueError(f"Unsupported llama_server generation settings: {sorted(set(generate) - supported)}")
    limit = generate.pop("max_tokens", -1)
    if not isinstance(limit, int) or limit == 0 or limit < -1:
        raise ValueError("max_tokens must be -1 (unlimited) or a positive integer")
    generate.pop("echo", None)
    payload = {**generate, "prompt": req["prompt"], "n_predict": limit,
               "stream": False, "cache_prompt": False, "n_keep": -1}
    if req.get("reasoning") and not req.get("messages"):
        raise ValueError("Explicit reasoning requires a chat-format prompt so the model template can enable it")
    result["runtime"]["server_binary"] = str(Path(command[0]).resolve())
    result["runtime"]["server_sha256"] = file_digest(Path(command[0]))
    result["runtime"]["server_version"] = subprocess.check_output(
        [command[0], "--version"], stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    result["runtime"]["server_command"] = command
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, data=None, timeout=None):
        body = json.dumps(data).encode("utf-8") if data is not None else None
        query = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                       headers={"Content-Type": "application/json"})
        with opener.open(query, timeout=timeout) as response:
            return json.load(response)

    server = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited {server.returncode} during startup; see worker log")
            try:
                if request("/health", timeout=1).get("status") == "ok":
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("llama-server was not ready within 180 seconds")
            time.sleep(0.1)
        if req.get("messages"):
            template_request = {"messages": req["messages"]}
            if req.get("reasoning") is not None:
                template_request["chat_template_kwargs"] = {"enable_thinking": req["reasoning"]}
            payload["prompt"] = request("/apply-template", template_request)["prompt"]
        result["rendered_prompt"] = payload["prompt"]
        result["prompt_tokens"] = request("/tokenize", {
            "content": payload["prompt"], "add_special": True, "parse_special": True})["tokens"]
        for _ in range(req["repeats"]):
            started = time.monotonic()
            completion = request("/completion", payload)
            stop_type = completion.get("stop_type")
            if stop_type not in {"eos", "word", "limit"}:
                raise RuntimeError(f"Unexpected llama-server stop type: {stop_type!r}")
            result["trials"].append({
                "text": completion["content"],
                "finish_reason": "length" if stop_type == "limit" else "stop",
                "context_shifted": completion.get("truncated", False),
                "usage": {"prompt_tokens": completion.get("tokens_evaluated"),
                          "completion_tokens": completion.get("tokens_predicted")},
                "seconds": time.monotonic() - started, "server_response": completion,
            })
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def worker(request_path: Path, result_path: Path):
    """Fresh worker per trial, or one model instance for all warm repetitions."""
    req = json.loads(request_path.read_text(encoding="utf-8"))
    result = {"runtime": fingerprint(), "trials": []}
    try:
        if req["backend"] == "demo":
            result["runtime"]["backend"] = "SIMULATED; not model evidence"
            for i in range(req["repeats"]):
                text = "Simulated baseline response."
                if req["generate"].get("temperature", 0) > 0:
                    text = "Simulated changed response."
                if req["load"].get("demo_unstable"):
                    text = f"Simulated variable response {req['trial_offset'] + i}."
                result["trials"].append({"text": text, "finish_reason": "stop", "usage": {}, "seconds": 0})
        elif req["backend"] == "llama_server":
            server_generate(req, result)
        else:
            import llama_cpp
            from llama_cpp import Llama
            result["runtime"]["llama_cpp_python"] = importlib.metadata.version("llama-cpp-python")
            result["runtime"]["dependencies"] = {
                name: importlib.metadata.version(name)
                for name in ("numpy", "diskcache", "Jinja2", "MarkupSafe", "typing_extensions")
            }
            info = llama_cpp.llama_print_system_info()
            result["runtime"]["llama_system_info"] = info.decode() if isinstance(info, bytes) else str(info)
            # Hash native libraries as well as recording their reported version.
            pkg = Path(llama_cpp.__file__).parent
            result["runtime"]["native_libraries"] = {
                str(p.relative_to(pkg)): file_digest(p)
                for p in sorted(pkg.rglob("*"))
                if p.is_file() and p.suffix in {".so", ".dylib", ".dll"}
            }
            llm = Llama(model_path=req["model"], **{"verbose": False, **req["load"]})
            # Record the actual prompt tokens, including the tokenizer's BOS behavior.
            result["prompt_tokens"] = llm.tokenize(req["prompt"].encode("utf-8"), special=True)
            for _ in range(req["repeats"]):
                llm.reset()
                started = time.monotonic()
                completion = llm.create_completion(prompt=req["prompt"], **req["generate"])
                choice = completion["choices"][0]
                result["trials"].append({
                    "text": choice["text"],
                    "finish_reason": choice["finish_reason"],
                    "usage": completion.get("usage", {}),
                    "seconds": time.monotonic() - started,
                })
    except Exception:
        result["error"] = traceback.format_exc()
    write_json(result_path, result)
    return 1 if "error" in result else 0


def summarize(records):
    rows = []
    baseline = {}
    for record in records:
        if record["variant"] == "baseline":
            baseline.setdefault(record["prompt"], record)
    for variant, prompt in dict.fromkeys((r["variant"], r["prompt"]) for r in records):
        trials = [r for r in records if r["variant"] == variant and r["prompt"] == prompt]
        ok = [r for r in trials if "error" not in r]
        ref = baseline.get(prompt, {})
        reference_hash = ref.get("text_sha256")
        hashes = {r["text_sha256"] for r in ok}
        rows.append({
            "variant": variant, "prompt": prompt, "attempts": len(trials),
            "successful": len(ok), "errors": len(trials) - len(ok),
            "unique_outputs": len(hashes),
            "repeatable_observed": len(ok) >= 2 and len(ok) == len(trials) and len(hashes) == 1,
            "matches_baseline": sum(r["text_sha256"] == reference_hash for r in ok) if reference_hash else None,
            "truncated": sum(r.get("finish_reason") == "length" for r in ok),
            "context_shifted": sum(bool(r.get("context_shifted")) for r in ok),
        })
    return rows


def markdown_report(report, out: Path):
    lines = ["# Output stability experiment", "", f"Backend: `{report['backend']}`. Mode: `{report['process_mode']}`.", "",
             "Exact UTF-8 response equality, including emitted reasoning and whitespace. Observations are limited to these trials.", "",
             "| Variant | Prompt | Successes / attempts | Unique outputs | Repeated identically | Matches first baseline | Hit token limit | Context shifted |",
             "|---|---|---:|---:|---|---:|---:|---:|"]
    for row in report["summary"]:
        esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {esc(row['variant'])} | {esc(row['prompt'])} | {row['successful']}/{row['attempts']} | {row['unique_outputs']} | {'yes' if row['repeatable_observed'] else 'no / insufficient'} | {row['matches_baseline'] if row['matches_baseline'] is not None else 'unavailable'} | {row['truncated']} | {row['context_shifted']} |")
    lines += ["", "A variant can repeat perfectly while consistently differing from baseline. Matching a truncated response does not establish full-answer stability.", "", "## Differences", ""]
    refs = {}
    for r in report["records"]:
        if r["variant"] == "baseline":
            refs.setdefault(r["prompt"], r)
    for r in report["records"]:
        if "error" in r:
            lines.append(f"- Error in trial {r['id']}; see `{r['artifact']}` and its worker log.")
            continue
        ref = refs.get(r["prompt"])
        if ref and "error" not in ref and r["text_sha256"] != ref["text_sha256"]:
            before, after = ref["text"], r["text"]
            offset = next((i for i, (a, b) in enumerate(zip(before, after)) if a != b), min(len(before), len(after)))
            diff_path = out / "artifacts" / f"{r['id']}.diff"
            diff_path.write_text("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="first baseline", tofile=r["id"])), encoding="utf-8")
            lines.append(f"- Trial {r['id']}: first difference at character {offset}; [diff](artifacts/{r['id']}.diff), [response]({r['artifact']}).")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    variants = [{"name": "baseline"}] + config.get("variants", [])
    settings = [(v, effective(config, v)) for v in variants]
    models = {s["model"] for _, s in settings if s["backend"] != "demo"}
    print("Fingerprinting model files...", flush=True)
    hashes = {m: file_digest(Path(m)) for m in sorted(models)}
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": config.get("backend", "llama_cpp"),
        "process_mode": config.get("process_mode", "fresh"),
        "runner": fingerprint(), "model_sha256": hashes,
        "config": copy.deepcopy(config), "records": [], "summary": [],
    }
    write_json(out / "report.json", report)
    repeats = config.get("repeats", 3)
    trial_count = 0
    for variant, setting in settings:
        for prompt in config["prompts"]:
            warm = report["process_mode"] == "warm"
            for offset in ([0] if warm else range(repeats)):
                trial_count += 1
                job_id = f"job-{trial_count:04d}"
                req_path = out / "artifacts" / f"{job_id}.request.json"
                res_path = out / "artifacts" / f"{job_id}.result.json"
                log_path = out / "artifacts" / f"{job_id}.log"
                req = {**setting, "prompt": prompt["text"], "repeats": repeats if warm else 1, "trial_offset": offset}
                if "messages" in prompt:
                    req["messages"] = prompt["messages"]
                write_json(req_path, req)
                print(f"{variant['name']} / {prompt['name']} / {'warm repeats' if warm else f'trial {offset + 1}/{repeats}'}", flush=True)
                failure = None
                try:
                    with log_path.open("wb") as log:
                        process = subprocess.Popen(
                            [setting["python"], str(Path(__file__).resolve()), "_worker", str(req_path), str(res_path)],
                            stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, **{str(k): str(v) for k, v in setting["env"].items()}},
                        )
                        try:
                            process.wait(timeout=config.get("timeout_seconds", 1800) or None)
                        finally:
                            if process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait()
                    if process.returncode:
                        failure = f"Worker exited {process.returncode}; see {log_path.name}"
                except (subprocess.TimeoutExpired, OSError) as exc:
                    failure = str(exc)
                result = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
                for i in range(req["repeats"]):
                    rec_id = f"{job_id}-{i + 1}"
                    record = {
                        "id": rec_id, "variant": variant["name"], "prompt": prompt["name"],
                        "prompt_sha256": prompt["sha256"], "repeat": offset + i + 1,
                        "settings": setting, "runtime": result.get("runtime", {}),
                        "prompt_tokens": result.get("prompt_tokens"),
                        "rendered_prompt": result.get("rendered_prompt"),
                        "artifact": f"artifacts/{rec_id}.json",
                    }
                    if i < len(result.get("trials", [])):
                        record.update(result["trials"][i])
                        record["text_sha256"] = digest(record["text"].encode("utf-8"))
                        (out / "artifacts" / f"{rec_id}.txt").write_bytes(record["text"].encode("utf-8"))
                    else:
                        record["error"] = result.get("error", failure or "Worker returned no trial")
                    write_json(out / record["artifact"], record)
                    report["records"].append(record)
                # Preserve partial work on interruption or a later failure.
                report["summary"] = summarize(report["records"])
                write_json(out / "report.json", report)
                markdown_report(report, out)
    print(f"Report: {out / 'report.md'}", flush=True)
    for row in report["summary"]:
        print(f"  {row['variant']} / {row['prompt']}: "
              f"{row['unique_outputs']} unique, {row['errors']} errors, "
              f"{row['matches_baseline']}/{row['successful']} match baseline, "
              f"{row['truncated']} hit token limit", flush=True)
    return report


def compare(left_path: Path, right_path: Path):
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    keys = lambda report: {(r["variant"], r["prompt"]) for r in report["records"]}
    rows = []
    for variant, prompt in sorted(keys(left) | keys(right)):
        a = [r for r in left["records"] if (r["variant"], r["prompt"]) == (variant, prompt)]
        b = [r for r in right["records"] if (r["variant"], r["prompt"]) == (variant, prompt)]
        ah = {r["text_sha256"] for r in a if "error" not in r}
        bh = {r["text_sha256"] for r in b if "error" not in r}
        same_prompt = bool(a and b) and {r["prompt_sha256"] for r in a} == {r["prompt_sha256"] for r in b}
        complete = len(a) >= 2 and len(b) >= 2 and all("error" not in r for r in a + b)
        same = complete and same_prompt and len(ah) == len(bh) == 1 and ah == bh
        rows.append({"variant": variant, "prompt": prompt, "identical_observed": same,
                     "same_prompt": same_prompt, "complete": complete,
                     "left_unique": len(ah), "right_unique": len(bh)})
    return {"left": str(left_path), "right": str(right_path), "comparisons": rows,
            "all_identical_observed": bool(rows) and all(r["identical_observed"] for r in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run_p = subs.add_parser("run", help="Run a local GGUF experiment")
    run_p.add_argument("config", type=Path)
    run_p.add_argument("--model", help="Override baseline model path (relative to current directory)")
    run_p.add_argument("--repeats", type=int)
    run_p.add_argument("--variant", action="append", help="Run only this named variant plus baseline; repeat to select more")
    run_p.add_argument("--prompt", action="append", help="Run only this named prompt; repeat to select more")
    run_p.add_argument("--out", type=Path)
    run_p.add_argument("--fail-on-change", action="store_true")
    run_p.add_argument("--max-tokens", type=int, help="Override all variants; -1 means unlimited for llama_server")
    demo_p = subs.add_parser("demo", help="Exercise reporting with SIMULATED outputs; no model needed")
    demo_p.add_argument("--out", type=Path)
    cmp_p = subs.add_parser("compare", help="Compare matching prompt/variant groups across report.json files")
    cmp_p.add_argument("left", type=Path)
    cmp_p.add_argument("right", type=Path)
    cmp_p.add_argument("--out", type=Path)
    cmp_p.add_argument("--fail-on-change", action="store_true")
    worker_p = subs.add_parser("_worker", help=argparse.SUPPRESS)
    worker_p.add_argument("request", type=Path)
    worker_p.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "_worker":
            # Allow the worker's finally block to stop its server on timeout or cancellation.
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
            return worker(args.request, args.result)
        if args.command == "compare":
            result = compare(args.left, args.right)
            print(json.dumps(result, indent=2))
            if args.out:
                write_json(args.out, result)
            return 2 if args.fail_on_change and not result["all_identical_observed"] else 0
        if args.command == "demo":
            config = {"backend": "demo", "repeats": 3,
                      "prompts": [{"name": "demo", "text": "demo", "sha256": digest(b"demo")}],
                      "variants": [{"name": "different-but-repeatable", "generate": {"temperature": 0.6}},
                                   {"name": "unstable", "load": {"demo_unstable": True}}]}
        else:
            config = read_config(args.config, args.model, args.repeats)
            if config.get("backend", "llama_cpp") not in {"llama_cpp", "llama_server"}:
                raise ValueError("run requires llama_cpp or llama_server; use demo for simulated data")
            if args.max_tokens is not None:
                config.setdefault("generate", {})["max_tokens"] = args.max_tokens
                for variant in config.get("variants", []):
                    variant.get("generate", {}).pop("max_tokens", None)
            for selected, field in ((args.variant, "variants"), (args.prompt, "prompts")):
                if selected:
                    available = {item["name"] for item in config.get(field, [])}
                    if set(selected) - available:
                        raise ValueError(f"Unknown {field}: {sorted(set(selected) - available)}")
                    config[field] = [item for item in config[field] if item["name"] in selected]
        out = (args.out or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")).resolve()
        report = run(config, out)
        if any("error" in r for r in report["records"]):
            return 1
        changed = any(not r["repeatable_observed"] or r["matches_baseline"] != r["successful"] for r in report["summary"])
        return 2 if getattr(args, "fail_on_change", False) and changed else 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
