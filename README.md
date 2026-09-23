# model-router

A small, local LLM router. Before a prompt is sent to an LLM, a local classifier
([Laya](https://huggingface.co/convaiinnovations/laya)) decides whether the prompt is
**simple** or **complex**. Simple prompts go to a cheap model, complex prompts to a
strong one. All LLM traffic flows through [agentgateway](https://agentgateway.dev),
which holds the API key and logs usage, cost and (optionally) prompts.

```
                 ┌──────────────── your machine ────────────────┐
                 │                                              │
  prompt ──────▶ │  app.py ──▶ Laya (local, ~150 ms, $0)        │
                 │     │        P(complex) >= 0.5 ?             │
                 │     │          no  → gpt-6-luna              │
                 │     │          yes → gpt-6-sol               │
                 │     ▼                                        │
                 │  agentgateway :4000  (adds API key, logs)    │ ──▶ OpenAI API
                 └──────────────────────────────────────────────┘
```

| Model | Role | Price per 1M tokens (input / output) |
|---|---|---|
| `gpt-6-luna` | simple prompts | $0.10 / $0.50 |
| `gpt-6-sol` | complex prompts | $2.00 / $10.00 (20× luna) |

Prices come from agentgateway's model catalog (`~/.config/agentgateway/base-costs.json`).

---

## The routing model: Laya

[Laya](https://huggingface.co/convaiinnovations/laya) is an open-weights (Apache-2.0)
**non-autoregressive decision model**. It does not generate text. You give it a *state*
(any text or JSON) and one or more *typed questions*, and it returns an answer with a
probability for every option in a single forward pass.

| | |
|---|---|
| Checkpoint | `convaiinnovations/laya` (English) |
| Backbone | ModernBERT-large encoder + small decision head, 421M parameters |
| Download | ~840 MB, cached in `~/.cache/huggingface/hub/` |
| Context | 512 tokens (longer prompts are truncated) |
| Hardware | CPU is fine; ~150 ms per decision on a laptop CPU, ~2 s to load |
| Question types | `choice` (pick one option), `score` (ordered scale), `noul` (yes/no) |
| Python package | [`laya`](https://pypi.org/project/laya/) 0.3.11 |

The router asks Laya one `choice` question:

```python
{"complexity": {
    "type": "choice",
    "instructions": "How complex is this prompt to answer well?",
    "criteria": {
        "simple":  "short factual lookup, greeting, trivial rewrite, one-step answer",
        "complex": "multi-step reasoning, math proofs, code design, deep analysis, long planning",
    },
}}
```

and routes on `probabilities["complex"]`. Laya was not trained specifically for
complexity grading; it answers this question zero-shot from the option descriptions.
Editing those descriptions in `app.py` is the easiest way to change routing behaviour.

---

## Getting started

Tested on Linux x86_64 with Python 3.12. You need an OpenAI API key with access to
`gpt-6-luna` and `gpt-6-sol`.

### 1. Install agentgateway

Download the binary from the
[GitHub releases](https://github.com/agentgateway/agentgateway/releases) (v1.5.0 used here):

```bash
# with the GitHub CLI
gh release download v1.5.0 -R agentgateway/agentgateway -p 'agentgateway-linux-amd64*'
sha256sum agentgateway-linux-amd64 && cat agentgateway-linux-amd64.sha256   # hashes must match
install -m 755 agentgateway-linux-amd64 ~/.local/bin/agentgateway            # any dir on your PATH
agentgateway --version
```

Use `agentgateway-linux-arm64` or `agentgateway-darwin-arm64` on other platforms.

### 2. Configure the models in agentgateway

Run `agentgateway` once and stop it (Ctrl+C). It creates its default config at
`~/.config/agentgateway/config.yaml`, which it loads whenever it is started without `-f`.

Open that file and replace `models: []` under `llm:` with:

```yaml
llm:
  gateways: default
  models:
  - name: gpt-6-luna
    provider: openAI
    params:
      model: gpt-6-luna
      apiKey: "$OPENAI_API_KEY"   # read from the environment, or paste the key here
  - name: gpt-6-sol
    provider: openAI
    params:
      model: gpt-6-sol
      apiKey: "$OPENAI_API_KEY"
```

If you use `$OPENAI_API_KEY`, export it in the shell that starts agentgateway. If you
paste the key into the file instead, run `chmod 600 ~/.config/agentgateway/config.yaml`.

Validate and start:

```bash
agentgateway -f ~/.config/agentgateway/config.yaml --validate-only
agentgateway
```

The gateway serves an OpenAI-compatible API on `http://localhost:4000/v1` and an admin
UI on <http://localhost:15000/ui>. It picks up config file changes without a restart.
Quick check:

```bash
curl localhost:4000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"gpt-6-luna","messages":[{"role":"user","content":"hi"}]}'
```

### 3. Install the Python side (Laya)

```bash
git clone <this repo> && cd model-router
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pulls the CPU-only build of PyTorch (no CUDA download). The Laya
weights (~840 MB) are downloaded from Hugging Face automatically the first time
`laya.load()` runs; no account or token is needed.

### 4. Run the router

With agentgateway running:

```bash
.venv/bin/python app.py --dry-run     # routing decisions only, no API calls, free
.venv/bin/python app.py               # built-in demo: 5 simple + 5 complex prompts
```

---

## What `app.py` does

For every prompt:

1. **Classify.** Laya scores the prompt and returns `P(complex)`.
2. **Pick a model.** `P(complex) >= threshold` (default 0.5) → `gpt-6-sol`, otherwise `gpt-6-luna`.
3. **Call the model** through agentgateway (`POST /v1/chat/completions`). The app never
   sees the OpenAI key; the gateway adds it.
4. **Report** Laya's score and latency, the chosen model, response time, tokens, cost, and
   the first line of the answer.

At the end it prints the total routed cost next to an estimate of what sending everything
to `gpt-6-sol` would have cost.

### Options

```bash
.venv/bin/python app.py "What is 2+2?" "Design a sharded job queue"   # your own prompts
.venv/bin/python app.py --threshold 0.4    # send more borderline prompts to gpt-6-sol
.venv/bin/python app.py --show             # print full answers
.venv/bin/python app.py --dry-run          # routing only
```

| Variable | Default | Purpose |
|---|---|---|
| `GATEWAY_URL` | `http://localhost:4000/v1/chat/completions` | agentgateway endpoint |

To route to different models, change `SMALL`, `LARGE` and `RATES` at the top of `app.py`
and add the models to the agentgateway config.

### Example output

```
[1] What is the capital of France?
    laya: P(complex)=0.11 in 149 ms -> gpt-6-luna
    gpt-6-luna: 2.4s, 13+16 tokens, $0.000009
    Paris.

[6] Design a multi-region rate limiter for an API with 50k req/s. Compare token bucket vs slid...
    laya: P(complex)=0.72 in 175 ms -> gpt-6-sol
    gpt-6-sol: 23.7s, 41+1308 tokens, $0.013162
    ## Recommended design
```

---

## How well does it route?

On the built-in demo set, all 10 prompts were routed as intended: simple prompts scored
0.04 to 0.26, complex prompts 0.56 to 0.73. A second set of harder prompts showed where
it works and where it doesn't:

| Case | Examples | Result |
|---|---|---|
| Long but easy | fix typos in a paragraph, list days/months/planets | correct (0.05 to 0.20 → luna) |
| Dutch / French | "Wat is de hoofdstad van België?", event-driven bank architecture | correct both ways |
| Short but hard | prove √2 irrational, Monty Hall, "Is P = NP?" | **0.42 to 0.52, coin flip** |
| Short trivial code | refactor a one-line list comprehension | 0.53 → sol (over-routed) |

### Known limitations

- **Grey zone.** Prompts that are short but need reasoning land near 0.5, where small
  wording changes flip the decision. Lowering `--threshold` to ~0.4 sends them to the
  strong model at little extra cost.
- **No evaluation set yet.** The accuracy above comes from ~20 hand-picked prompts.
  Build a labelled set from real traffic before trusting the threshold.
- **Only the prompt is scored.** Multi-turn history, system prompts and attachments are
  ignored, and anything past 512 tokens is truncated.
- **Routing lives in the client.** Every app embeds Laya and PyTorch. A shared routing
  service in front of the gateway would give all clients routing through one `auto` model.
- **No fallback.** If `gpt-6-luna` fails or answers poorly, nothing retries on `gpt-6-sol`.
- **Savings depend on your traffic.** Simple prompts are cheap on either model; real
  savings come from medium-difficulty prompts moving to luna. The demo (50% complex)
  saved only 2%. The "all gpt-6-sol" figure is an estimate that reuses luna's token counts.
- Laya prints a `RuntimeWarning` about uncalibrated temperatures on load. It concerns
  questions with 11+ options and does not affect this 2-option question.

---

## Observability

agentgateway records every request (model, tokens, latency, cost) in
`~/.config/agentgateway/data.db`, viewable in the admin UI. To also store full prompt
and completion text, add to the config (or toggle it in the UI):

```yaml
frontendPolicies:
  accessLog:
    database:
      llm: full        # "metadata" stores usage and cost without prompt text
```

Stored prompts are plain text on disk.

## Security notes

- agentgateway listens on all interfaces on port 4000 with **no authentication**. Anyone
  who can reach the port can spend your API credit. Firewall it, or add auth before
  exposing it.
- Keep API keys out of this repo; `.env` is git-ignored.

---

## Repository layout

| Path | What it is |
|---|---|
| `app.py` | The router demo described above |
| `bench.py` | Latency benchmark for Laya on an email-triage example (1 vs 3 questions) |
| `requirements.txt` | Python dependencies (CPU PyTorch, laya, requests, python-dotenv) |
| `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` | Fine-tuning Laya on the `typed-decisions` dataset on Kaggle's 2×T4 GPUs; a starting point for training a dedicated complexity classifier |
| `reports/laya-report.html` | Laya vs Jev benchmark report on five decision tasks |

## Credits

- [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations, Apache-2.0
- [agentgateway](https://github.com/agentgateway/agentgateway), Apache-2.0
- Benchmark comparisons build on [sysone-bench](https://github.com/instax-dutta/sysone-bench)
