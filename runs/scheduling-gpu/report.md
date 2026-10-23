# Output stability experiment

Backend: `llama_server`. Mode: `warm`.

Exact UTF-8 response equality, including emitted reasoning and whitespace. Observations are limited to these trials.

| Variant | Prompt | Successes / attempts | Unique outputs | Repeated identically | Matches first baseline | Hit token limit | Context shifted |
|---|---|---:|---:|---|---:|---:|---:|
| baseline | scheduling | 4/4 | 1 | yes | 4 | 0 | 0 |
| threads-4 | scheduling | 4/4 | 1 | yes | 4 | 0 | 0 |

A variant can repeat perfectly while consistently differing from baseline. Matching a truncated response does not establish full-answer stability.

## Differences

