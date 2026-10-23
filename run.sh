python3 stability_lab.py run experiment.json \
  --model /home/daniel/.local/share/drove/models/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M/gemma-4-E4B-it-Q4_K_M.gguf \
  --prompt scheduling --variant threads-4 --repeats 4 \
  --out runs/scheduling-gpu
