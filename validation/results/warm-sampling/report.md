# Output stability experiment

Backend: `llama_cpp`. Mode: `warm`.

Exact UTF-8 response equality, including emitted reasoning and whitespace. Observations are limited to these trials.

| Variant | Prompt | Successes / attempts | Unique outputs | Repeated identically | Matches first baseline | Hit token limit |
|---|---|---:|---:|---|---:|---:|
| baseline | arithmetic | 2/2 | 1 | yes | 2 | 2 |

A variant can repeat perfectly while consistently differing from baseline. Matching a truncated response does not establish full-answer stability.

## Differences

