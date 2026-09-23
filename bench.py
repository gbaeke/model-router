import time, statistics, laya

t0 = time.perf_counter()
agent = laya.load("convaiinnovations/laya")
print(f"load: {time.perf_counter() - t0:.1f}s")

state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}
single = {"churn_risk": questions["churn_risk"]}

r = agent.predict(state, questions)  # warm-up
for k, v in r["answers"].items():
    print(k, v)

def bench(qs, n=20):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); agent.predict(state, qs); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)

print(f"single question p50: {bench(single):.1f} ms")
print(f"3 questions p50:     {bench(questions):.1f} ms")
