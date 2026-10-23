# Output stability experiment

Backend: `llama_cpp`. Mode: `fresh`.

Exact UTF-8 response equality, including emitted reasoning and whitespace. Observations are limited to these trials.

| Variant | Prompt | Successes / attempts | Unique outputs | Repeated identically | Matches first baseline | Hit token limit |
|---|---|---:|---:|---|---:|---:|
| baseline | arithmetic | 2/2 | 1 | yes | 2 | 2 |
| threads-4 | arithmetic | 2/2 | 1 | yes | 2 | 2 |
| seed-43-greedy | arithmetic | 2/2 | 1 | yes | 2 | 2 |
| temperature-0.6 | arithmetic | 2/2 | 1 | yes | 0 | 2 |

A variant can repeat perfectly while consistently differing from baseline. Matching a truncated response does not establish full-answer stability.

## Differences

- Trial job-0007-1: first difference at character 28; [diff](artifacts/job-0007-1.diff), [response](artifacts/job-0007-1.json).
- Trial job-0008-1: first difference at character 28; [diff](artifacts/job-0008-1.diff), [response](artifacts/job-0008-1.json).
