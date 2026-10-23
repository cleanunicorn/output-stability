# Validation performed

Eight automated tests cover exact whitespace comparison, repeatability versus baseline agreement, errors, truncated output, changed/missing prompts, partial reports, fresh/warm subprocess execution, worker launch failures, and input preservation. All passed.

Real inference was also tested with Qwen3-0.6B-Q8_0 on Linux x86-64, an AMD EPYC 9V74 CPU, Python 3.12.14, and llama-cpp-python 0.3.16 built with GCC. Full runtime, native-library, and model fingerprints are in the saved JSON reports.

| Configuration | Repetitions | Distinct outputs | Matches greedy baseline |
|---|---:|---:|---:|
| CPU, 1 thread, temperature 0, seed 42 | 2 | 1 | 2/2 |
| CPU, 4 inference threads | 2 | 1 | 2/2 |
| Greedy, seed 43 | 2 | 1 | 2/2 |
| Temperature 0.6, seed 42, fresh processes | 2 | 1 | 0/2 |
| Temperature 0.6, seed 42, warm instance | 2 | 1 | 0/2 |

The warm sampling responses also matched the fresh-process sampling responses exactly. These are **128-token prefix checks on one prompt**, not full-answer or universal determinism claims. All ten generations hit that token limit.

See [fresh-process report](results/fresh/report.md) and [warm sampling report](results/warm-sampling/report.md) for raw responses, diffs, and trial evidence. These result files are historical evidence from this workspace; absolute model/interpreter paths inside them refer to that workspace.

Reproduce with:

```bash
python stability_lab.py run validation/smoke.json --out runs/my-smoke
python stability_lab.py run validation/warm-sampling.json --out runs/my-warm
```

Both configurations expect `models/Qwen3-0.6B-Q8_0.gguf`; use `--model` to override it. Different machines may produce different results, which is what the project is designed to investigate.

`tested-requirements.txt` records the exact Python dependencies used here. It is a reference for this Python 3.12 environment, not a cross-platform lockfile. Rebuilding the same backend version with another compiler or on another CPU may produce a different native library.

The workspace initially pointed `CC` at an unavailable Clang executable. The successful Linux installation selected the installed GCC explicitly:

```bash
CC=gcc CXX=g++ CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install -e '.[inference]'
```

This workaround is only needed when the default compiler selection is broken.
