"""GPT-5 as the server recommender (the paper's main results), through the OpenAI Batch API.

Reads the caches of scripts/run_pipeline.py in --out (data.pkl, bert.pkl). For each job the server receives the full
history (full), the history without the three health and beauty categories (cat), or the history without the products
the BERT obfuscator withholds (bert), and is always asked for N = 10 recommendations; the framework keeps the first n_ns
(scripts/evaluate.py). Only product lines are sent; each request carries an index (u0000, ...) in the order of the users
in data.pkl, never a user id.
Stages:
  prepare JOB   write gpt5/requests_JOB.jsonl (skipping requests already answered) and print its worst-case cost
  submit JOB    upload it to the Batch API; refused if the worst case, with what was spent before, exceeds --budget
  collect JOB   when the batch has finished, add its answers to gpt5/answers_JOB.jsonl
  build         parse the answers of every job -> server_recs_gpt5.pkl, the input of scripts/evaluate.py --server gpt5
submit and collect need OPENAI_API_KEY. The answers of the paper's run are in results/gpt5/answers_*.jsonl; copy them to
<out>/gpt5/ and run build to evaluate without calling the API.
Example: python scripts/gpt5_server.py prepare bert --out runs/main"""
import argparse, glob, json, os, pickle, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from privrec import data, prompts, llm, obfuscators as ob

MODEL, EFFORT, N = "gpt-5-2025-08-07", "low", 10
MAX_TOK = {"full": 8000, "cat": 5000, "bert": 5000}  # reasoning + visible output; the longest answer in the paper's run used 4,699
PRICE = (0.625, 5.0)  # Batch API, USD per 1M input / output tokens

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["prepare", "submit", "collect", "build"])
ap.add_argument("job", nargs="?", choices=["full", "cat", "bert"])
ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=float, default=100.0, help="USD limit over all batches in --out")
args = ap.parse_args()
if args.stage != "build" and not args.job: sys.exit("give the job: full, cat or bert")
G = os.path.join(args.out, "gpt5"); os.makedirs(G, exist_ok=True)
F = lambda kind, job=None: os.path.join(G, f"{kind}_{job or args.job}." + ("txt" if kind in ("batch_id", "collected") else "jsonl"))
D = pickle.load(open(os.path.join(args.out, "data.pkl"), "rb")); users, H, M = D["users"], D["H"], D["meta"]
ids = {f"u{i:04d}": u for i, u in enumerate(users)}


def read_jsonl(f): return [json.loads(x) for x in open(f)] if os.path.exists(f) else []


def sent(job):
    """{user: products sent to the server} for a job."""
    if job == "full": return H
    f = ob.categorical(M) if job == "cat" else ob.bert(pickle.load(open(os.path.join(args.out, "bert.pkl"), "rb")))
    return {u: ob.split(H[u], f)[0] for u in users}


def worst(f, job):  # every request uses its whole token allowance (~3 characters per input token)
    return sum(len(r["body"]["messages"][0]["content"]) / 3 * PRICE[0] + MAX_TOK[job] * PRICE[1] for r in read_jsonl(f)) / 1e6


def prepare():
    S, done = sent(args.job), {x["custom_id"] for x in read_jsonl(F("answers"))}
    hist = lambda xs: "\n".join(f"{i + 1}. {data.show(M[a])}" for i, a in enumerate(xs))
    with open(F("requests"), "w") as f:
        for c, u in ids.items():
            if S[u] and c not in done:
                f.write(json.dumps(dict(custom_id=c, method="POST", url="/v1/chat/completions", body=dict(
                    model=MODEL, messages=[dict(role="user", content=prompts.SERVER.format(n=N, h=hist(S[u])))],
                    reasoning_effort=EFFORT, max_completion_tokens=MAX_TOK[args.job]))) + "\n")
    print(f"{len(read_jsonl(F('requests')))} requests in {F('requests')}; worst case ${worst(F('requests'), args.job):.2f}")


def submit():
    from openai import OpenAI
    if os.path.exists(F("batch_id")): sys.exit(f"already submitted: {open(F('batch_id')).read()}")
    spent = sum((x["prompt_tokens"] * PRICE[0] + x["completion_tokens"] * PRICE[1]) / 1e6 for f in glob.glob(f"{G}/answers_*.jsonl") for x in read_jsonl(f))
    pending = sum(worst(F("requests", j), j) for j in MAX_TOK if j != args.job and os.path.exists(F("batch_id", j)) and not os.path.exists(F("collected", j)))
    new = worst(F("requests"), args.job)
    print(f"spent ${spent:.2f}, running batches at most ${pending:.2f}, this batch at most ${new:.2f}, budget ${args.budget:.2f}")
    if spent + pending + new > args.budget: sys.exit("refused: could exceed the budget")
    c = OpenAI(); fid = c.files.create(file=open(F("requests"), "rb"), purpose="batch").id
    b = c.batches.create(input_file_id=fid, endpoint="/v1/chat/completions", completion_window="24h")
    open(F("batch_id"), "w").write(b.id); print("submitted", b.id)


def collect():
    from openai import OpenAI
    c = OpenAI(); b = c.batches.retrieve(open(F("batch_id")).read().strip()); print(b.status, b.request_counts)
    if b.status not in ("completed", "cancelled", "expired") or not b.output_file_id: return
    A = {x["custom_id"]: x for x in read_jsonl(F("answers"))}
    for line in c.files.content(b.output_file_id).text.splitlines():
        x = json.loads(line); r = x.get("response") or {}
        if r.get("status_code") != 200: continue
        body = r["body"]; ch = body["choices"][0]
        A[x["custom_id"]] = dict(custom_id=x["custom_id"], content=ch["message"]["content"] or "", finish_reason=ch["finish_reason"],
                                 prompt_tokens=body["usage"]["prompt_tokens"], completion_tokens=body["usage"]["completion_tokens"])
    with open(F("answers"), "w") as f:
        for k in sorted(A): f.write(json.dumps(A[k]) + "\n")
    open(F("collected"), "w").write(b.id + "\n")
    X = list(A.values()); cost = sum(x["prompt_tokens"] * PRICE[0] + x["completion_tokens"] * PRICE[1] for x in X) / 1e6
    print(f"{len(X)} answers, {sum(x['finish_reason'] == 'length' for x in X)} cut off by the token limit, ${cost:.2f}")


def build():
    key = {"full": "full", "cat": "cat_obf", "bert": "bert_obf"}
    out = {key[j]: {ids[x["custom_id"]]: (x["content"], llm.parse(x["content"], N)) for x in read_jsonl(F("answers", j))}
           for j in key if os.path.exists(F("answers", j))}
    pickle.dump(out, open(os.path.join(args.out, "server_recs_gpt5.pkl"), "wb"))
    print({k: len(v) for k, v in out.items()}, "->", os.path.join(args.out, "server_recs_gpt5.pkl"))


dict(prepare=prepare, submit=submit, collect=collect, build=build)[args.stage]()
