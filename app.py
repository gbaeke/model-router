"""Small model router: Laya classifies each prompt's complexity locally, then the
prompt goes through agentgateway to gpt-6-luna (simple) or gpt-6-sol (complex)."""
import argparse, os, time

os.environ.setdefault("USE_TF", "0")

import laya, requests

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:4000/v1/chat/completions")
SMALL, LARGE = "gpt-6-luna", "gpt-6-sol"
# $ per 1M tokens (input, output), from agentgateway's base-costs.json
RATES = {SMALL: (0.1, 0.5), LARGE: (2.0, 10.0)}

QUESTION = {"complexity": {
    "type": "choice",
    "instructions": "How complex is this prompt to answer well?",
    "criteria": {
        "simple": "short factual lookup, greeting, trivial rewrite, one-step answer",
        "complex": "multi-step reasoning, math proofs, code design, deep analysis, long planning",
    },
}}

PROMPTS = [
    "What is the capital of France?",
    "Translate 'good morning' into Spanish.",
    "Give me a synonym for 'happy'.",
    "What is 12 times 8?",
    "Write a one-line birthday wish for a colleague.",
    "Design a multi-region rate limiter for an API with 50k req/s. Compare token bucket vs sliding window, "
    "explain the consistency tradeoffs, and sketch the data model.",
    "Prove that there are infinitely many primes, then explain how the proof changes for primes of the form 4k+3.",
    "Our Postgres queries got 10x slower after a migration that added a JSONB column and a GIN index. "
    "Walk through a step-by-step diagnosis plan and the most likely root causes.",
    "Write a Python function that parses a cron expression and returns the next 5 run times, handling ranges, steps and lists.",
    "Analyze the second-order economic effects of a 4-day work week on a mid-sized manufacturing company.",
]


def route(agent, prompt, threshold):
    t = time.perf_counter()
    ans = agent.predict({"prompt": prompt}, QUESTION)["answers"]["complexity"]
    ms = (time.perf_counter() - t) * 1000
    p_complex = ans["probabilities"]["complex"]
    return (LARGE if p_complex >= threshold else SMALL), p_complex, ms


def complete(model, prompt):
    t = time.perf_counter()
    r = requests.post(GATEWAY, json={"model": model, "messages": [{"role": "user", "content": prompt}]}, timeout=300)
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"]["content"], body["usage"], time.perf_counter() - t


def cost(model, usage):
    i, o = RATES[model]
    return (usage["prompt_tokens"] * i + usage["completion_tokens"] * o) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompts", nargs="*", help="prompts to route (default: built-in demo set)")
    ap.add_argument("--threshold", type=float, default=0.5, help="P(complex) at or above which gpt-6-sol is used")
    ap.add_argument("--dry-run", action="store_true", help="only show routing decisions, don't call the models")
    ap.add_argument("--show", action="store_true", help="print full model answers")
    args = ap.parse_args()

    print("loading laya...")
    agent = laya.load("convaiinnovations/laya")
    prompts = args.prompts or PROMPTS

    total = baseline = 0.0
    for n, prompt in enumerate(prompts, 1):
        model, p_complex, route_ms = route(agent, prompt, args.threshold)
        print(f"\n[{n}] {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
        print(f"    laya: P(complex)={p_complex:.2f} in {route_ms:.0f} ms -> {model}")
        if args.dry_run:
            continue
        text, usage, secs = complete(model, prompt)
        c = cost(model, usage)
        total += c
        baseline += cost(LARGE, usage)
        print(f"    {model}: {secs:.1f}s, {usage['prompt_tokens']}+{usage['completion_tokens']} tokens, ${c:.6f}")
        print("    " + (text if args.show else text.strip().splitlines()[0][:120]))

    if not args.dry_run:
        saved = (1 - total / baseline) * 100 if baseline else 0
        print(f"\nrouted cost ${total:.6f} vs all-{LARGE} ${baseline:.6f} ({saved:.0f}% saved)")


if __name__ == "__main__":
    main()
