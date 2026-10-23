# Output Stability Lab

Run a local open-weights reasoning model repeatedly, change parameters, and see exactly when its output changes. No response caching or cloud account. The runner uses Python's standard library and manages a local `llama-server` process on loopback. The optional `llama_cpp` backend remains available for historical experiments.

Two different questions are measured:

- **Repeatability:** do all repetitions of the same configuration produce identical response bytes?
- **Baseline agreement:** does a changed configuration still produce the first baseline response?

A configuration can be repeatable while consistently producing a different answer. Reports compare all emitted text, including reasoning and whitespace. They flag truncated generations separately.

## Included experiment artifacts

The repository includes the [Gemma GPU scheduling report](runs/scheduling-gpu/report.md), its [machine-readable results](runs/scheduling-gpu/report.json), and all associated requests, responses, and worker logs. Four baseline repetitions and four `threads-4` repetitions produced identical outputs with no errors or token-limit stops. These are observations for this model, prompt, and runtime, not a general determinism guarantee.

[Historical Qwen validation artifacts](validation/README.md) are also included. Saved reports retain machine-specific model/interpreter paths and runtime fingerprints; model weights and virtual environments are excluded. New runs are ignored by Git unless explicitly added. Choose a new output directory when reproducing an included experiment.

## Install

Use Python 3.10–3.12 for the simplest installation experience. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a local [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) and make `llama-server` available on PATH, or set `server_binary` to its absolute path in your experiment JSON. The supplied configurations use `"backend": "llama_server"`; no Python inference package is needed. The supplied configurations offload all model layers to GPU and enable Flash Attention. Set `n_gpu_layers: 0` for CPU-only experiments.

For the older Python backend, set `"backend": "llama_cpp"` and install `python -m pip install -e '.[inference]'`. Its optional dependency remains pinned to `llama-cpp-python==0.3.16`.

Download the small reasoning model used by the supplied prompts:

```bash
python -m pip install huggingface_hub
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
```

This is a roughly 600-million-parameter reasoning model: useful for testing on a laptop, with more limited reasoning quality than larger models. The GGUF is about 640 MB. Larger Qwen3 GGUF models also work. Downloading requires internet; experiments run locally after installation. Review the model's own license when choosing a model.

## First experiment

Start with one prompt, one parameter change, and two repetitions:

```bash
python stability_lab.py run experiment.json \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt arithmetic --variant threads-4 --repeats 2 --out runs/quick
```

Open `runs/quick/report.md`. The adjacent `report.json` contains machine-readable results. `artifacts/` contains exact response text, per-trial JSON, worker logs, and diffs against the first baseline response. Output directories must be new to avoid overwriting experiments.

Run the full supplied sweep:

```bash
python stability_lab.py run experiment.json \
  --model models/Qwen3-0.6B-Q8_0.gguf --out runs/cpu
```

The full sweep runs 63 generations. The supplied configurations use warm mode, loading once per variant/prompt and recomputing each response without prompt caching. Each job prints progress; reports are checkpointed after each job. Ctrl+C stops the run, retaining completed results. The supplied configurations set `timeout_seconds: 0`, disabling the worker deadline. Set a positive number of seconds to impose a deadline; in warm mode it covers all repetitions. Configurations that omit the field retain the 30-minute default.

The `scheduling` prompt contains the two-worker scheduling problem, including optimality proofs and delayed worker availability. Select it in either experiment configuration with `--prompt scheduling`. For example:

```bash
python stability_lab.py run experiment.json \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt scheduling --variant threads-4 --repeats 2 --out runs/scheduling
```

Both supplied configurations use `generate.max_tokens: -1`: generate until the model emits an end token or configured stop string. You can also pass `--max-tokens -1` to override every selected variant. A positive value opts into a token limit.

## What the supplied sweep changes

The baseline uses GPU offloading, Flash Attention, eight inference threads, twelve prompt-processing threads, greedy decoding, and seed 42. Reasoning is explicitly enabled with no reasoning-token budget. Every variant independently inherits the baseline and overrides only its listed fields:

| Variant | Change | What it investigates |
|---|---|---|
| `threads-4` | Inference threads: 8 → 4 | Parallel computation |
| `batch-threads-4` | Prompt-processing threads: 12 → 4 | Prompt evaluation |
| `context-8192` | Context: 32768 → 8192 | Context configuration |
| `temperature-0.6` | Temperature: 0 → 0.6 | Seeded sampling versus greedy decoding |
| `top-p-0.8` | Top-p: 0.95 → 0.8 | Expected to be irrelevant during greedy decoding |
| `seed-43-greedy` | Seed: 42 → 43 | Control: expected to be irrelevant during greedy decoding |

The seed stays fixed except in the deliberately named seed control. No result is assumed in advance. The prompt-processing thread setting is distinct from concurrent request batching; this project runs requests sequentially.

Unlimited generation removes the output-token cap; it does not guarantee that a model will eventually finish. The server enables context shifting and retains the original prompt (`n_keep: -1`), allowing supported models to continue beyond the context window while discarding older generated context. Reports count context shifts separately from token-limit stops and preserve the native server response. The default context is 32768 tokens. A model that cannot shift context may still stop or fail at its context limit. Ctrl+C stops the run and its managed server, retaining completed jobs.

## Customize experiments

Edit `experiment.json`. Relative model and prompt paths resolve against the configuration file's directory; `--model` resolves against the current working directory. The supplied prompts use `format: "chat"` and JSON message arrays (`prompts/*.chat.json`). The runner applies the loaded model's own chat template through `/apply-template`, explicitly sets `enable_thinking: true`, and saves the rendered prompt and actual input tokens. This supports Gemma without sending Qwen-specific control tokens. Generation still uses `/completion` to retain raw reasoning and answer text together. Set top-level `reasoning` to `false` to disable it. Legacy raw Qwen prompts remain available as `.txt` files; explicit reasoning requires chat format. Model templates may inject their own defaults; the saved rendered prompt is the record of what was actually sent.

For `llama_server`, `load` supports `n_ctx`, `n_threads`, `n_threads_batch`, `n_gpu_layers`, `n_batch`, `n_ubatch`, `use_mmap`, and `flash_attn`, mapped to native CLI flags. `generate.max_tokens` maps to native `n_predict`; sampling settings are sent to `/completion`. Unsupported settings produce an error. Requests use the rendered prompt and disable prompt caching. Server path, binary hash, version, launch arguments, and returned generation settings are recorded.

For `llama_cpp`, `load` and `generate` are passed to `Llama` and `create_completion`. Its token-limit semantics differ: use a positive budget for that backend. Backend defaults can differ, so inspect recorded effective server settings when comparing installations.

For sampling experiments, make the baseline temperature positive (for example, 0.6), then vary `top_p`, `top_k`, `min_p`, `repeat_penalty`, or `seed` one at a time. At temperature zero, sampling filters and the seed generally do not affect greedy selection.

`experiment-sampling.json` provides this second sweep, with a temperature-0.6 baseline and sampling parameters varied independently. Run it with the same command, substituting that configuration filename. Its `seed-43-sampling` variant is the explicitly named seed control; all other variants retain seed 42.

Add GPU or model variants only when the appropriate backend and files are available:

```json
{"name": "gpu-all", "load": {"n_gpu_layers": -1}}
```

```json
{"name": "other-quantization", "model": "models/another-quantization.gguf"}
```

A CPU-only build cannot test GPU execution. Confirm offloading in the worker logs; native server output is captured in those logs. Parameters such as `flash_attn` or `n_batch` can also be tested in `load`. A flag being accepted does not prove it changed the execution path. Quantization comparisons need the corresponding separately downloaded GGUF.

Each variant may also set `python` to an absolute interpreter path and `env` to explicit environment overrides. This lets you compare separate virtual environments or backend builds:

```json
{
  "name": "different-backend-build",
  "python": "/absolute/path/to/other-venv/bin/python",
  "env": {"OMP_NUM_THREADS": "1"}
}
```

The alternate interpreter needs `llama-cpp-python` only for the `llama_cpp` backend. It executes this project's worker script directly. Environment overrides are applied before starting the backend; only explicitly configured overrides are saved, not your full environment. Do not put secrets into experiment configurations.

## Fresh processes versus a loaded server

`"process_mode": "fresh"` (the fallback when omitted) loads the model for every trial. The supplied configurations select `"warm"` to avoid repeated loading. This checks repeatability across process restarts.

Set `"process_mode": "warm"` to load the model once per variant/prompt and repeat generation on that instance. The native backend disables prompt caching and passes the configured seed for every request. The Python backend resets the evaluated token count and does not install a `LlamaCache`. Neither reuses a stored response. Both modes recompute generation. Warm mode models sequential reuse of one model instance; it does not test a production server's scheduling or continuous batching.

Run both modes to explore whether process lifetime matters. Do not change multiple things at once if you want to attribute a difference to a specific setting.

## Compare machines, installations, or runs

Run the same experiment on each machine, copy the result directories, then:

```bash
python stability_lab.py compare runs/machine-a/report.json runs/machine-b/report.json \
  --out comparison.json
```

Comparison matches variant and prompt names, checks prompt hashes, and requires every successful response in both groups to be the same. Missing cases, trial errors, or variation within either run prevent a passing comparison. Inspect each report's configuration, runtime, and model fingerprints to identify what changed. Comparison does not claim that two configurations are equivalent merely because the observed text agrees.

Recorded evidence includes model SHA-256, complete effective settings, prompt text and hash, actual prompt token IDs, backend version, server binary hash and native response (native backend), native-library hashes and backend system information (Python backend), Python/platform information, finish reason, token counts, generation time, and exact response hashes. No generated-token-ID comparison is claimed: the primary comparison is the text returned by the backend, with configured stop strings omitted. GPU driver versions, compiler flags, full operating-system images, all environment variables, and external system load are not captured automatically. Save those separately when investigating them. Model hashing occurs once at the beginning; leave files unchanged for the whole run.

Repeat each setting more times and use more prompts for stronger evidence. Identical results from a finite experiment do not prove determinism for every possible prompt or future runtime. This tests output stability, not answer correctness, logit equality, or hidden internal state. Differences in latency, finish reason, and token counts remain visible in JSON even if returned response text matches.

## Validate the project without a model

```bash
python -m unittest discover -s tests -v
python stability_lab.py demo --out runs/demo
```

The demo runs **simulated** stable, consistently different, and unstable outputs through the real subprocess/reporting pipeline. It is explicitly labeled and is not evidence about a model.

The project also includes [real-model validation results](validation/README.md): ten Qwen3 generations covering fresh processes, warm sampling, thread count, seed, and temperature changes. Those deliberately short checks compare 128-token prefixes. Model weights and virtual environments are not included in the project archive.

Exit status is 0 for completed experiments, 1 for configuration or inference errors. Add `--fail-on-change` to `run` or `compare` to return 2 when observed outputs differ or stability is insufficient. Without this flag, discovering a difference is a normal successful experiment.

## Gemma scheduling on a local GPU

The supplied configurations enable model-template reasoning and unlimited generation. For example:

```bash
python3 stability_lab.py run experiment.json \
  --model /path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  --prompt scheduling --variant threads-4 --repeats 4 \
  --out runs/scheduling-gpu
```

This runs four baseline repetitions followed by four variant repetitions. Use a new output directory. GPU memory must fit the model and context; reduce `n_ctx` or offloaded layers if necessary. Gemma 4 in the tested local build does not support context shifting: unlimited tokens remove the generation cap, but the 32768-token context remains a boundary. Thread-count changes may have little effect when generation is GPU-bound.
