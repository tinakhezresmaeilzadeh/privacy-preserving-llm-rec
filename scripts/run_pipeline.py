"""End-to-end pipeline of the paper, fully local. Every stage is cached in --out, so an interrupted run resumes.

  data     histories, targets and catalogue from the Amazon Reviews 2023 files      -> data.pkl
  labels   independent sensitivity labels and scores (Qwen3.5-9B, Appendix A prompts) -> labels.pkl
  bert     P(sensitive) of the BERT obfuscator for every product                     -> bert.pkl
  server   server-side recommendations (Qwen3.5-9B) for every configuration         -> server_recs.pkl
  local    on-device recommendations for each deobfuscator                           -> local_recs_<name>.pkl
Then run scripts/evaluate.py --out <same dir>.

Example:
  python scripts/run_pipeline.py --data-dir amazon2023/ --users users.txt --bert-dir models/bert_obfuscator --out runs/main
"""
import argparse, os, pickle, re, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from privrec import data, prompts, llm, obfuscators as ob

N = 10
SERVER = "Qwen/Qwen3.5-9B"
LOCALS = {"Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct", "Qwen3.5-2B": "Qwen/Qwen3.5-2B", "Qwen3.5-4B": "Qwen/Qwen3.5-4B"}

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True, help="directory with <Category>.jsonl and meta_<Category>.jsonl")
ap.add_argument("--users", required=True, help="text file with one evaluation user_id per line")
ap.add_argument("--bert-dir", required=True, help="fine-tuned BERT obfuscator (scripts/train_obfuscator.py)")
ap.add_argument("--out", required=True)
ap.add_argument("--locals", default=",".join(LOCALS), help="comma-separated deobfuscators to run")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
log = lambda *a: print(*a, flush=True)


def cached(name, fn):
    f = os.path.join(args.out, f"{name}.pkl")
    if os.path.exists(f): return pickle.load(open(f, "rb"))
    x = fn(); pickle.dump(x, open(f, "wb")); return x


D = cached("data", lambda: data.build(args.data_dir, [l.strip() for l in open(args.users) if l.strip()]))
users, H, T, M = D["users"], D["H"], D["T"], D["meta"]
items = sorted({a for u in users for a in H[u]} | {T[u] for u in users})
show = lambda a: data.show(M[a])
hist = lambda xs: "\n".join(f"{i + 1}. {show(a)}" for i, a in enumerate(xs))
log("users", len(users), "products", len(items), "catalogue", len(M))


def labels():
    tok, m = llm.load(SERVER); c = lambda t: llm.chat(tok, t, enable_thinking=False)
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in ["sensitive", "nonsensitive"]]
    p = {}
    for i in range(0, len(items), 64):  # label = argmax of the first-token probabilities of the two labels
        b = items[i:i + 64]; enc = tok([c(prompts.LABEL + show(a)) for a in b], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad(): lg = m(**enc).logits[:, -1, ids].float()
        p.update(zip(b, torch.softmax(lg, -1)[:, 0].tolist()))
    raw = llm.generate(tok, m, [c(prompts.SCORE + show(a)) for a in items], 6, 64)
    num = lambda t: (lambda x: min(1.0, max(0.0, float(x[0]))) if x else np.nan)(re.findall(r"\d*\.?\d+", t))
    del m; torch.cuda.empty_cache()
    return dict(p_sens=p, score=dict(zip(items, map(num, raw))))
L = cached("labels", labels)

BP = cached("bert", lambda: ob.bert_probs(args.bert_dir, M, items))
OBF = {"cat": ob.categorical(M), "bert": ob.bert(BP)}
SPLIT = {o: {u: ob.split(H[u], f) for u in users} for o, f in OBF.items()}


def server_recs():
    tok, m = llm.load(SERVER); c = lambda t: llm.chat(tok, t, enable_thinking=False)
    jobs = {"full": {u: (H[u], N) for u in users}}
    for o in OBF:
        jobs[f"{o}_obf"] = {u: (SPLIT[o][u][0], N) for u in users if SPLIT[o][u][0]}  # obfuscation only: all N from the server
        jobs[f"{o}_srv"] = {u: (SPLIT[o][u][0], ob.counts(*SPLIT[o][u], N)[0]) for u in users
                            if SPLIT[o][u][0] and ob.counts(*SPLIT[o][u], N)[0]}  # framework: the server's n_ns
    out = {}
    for k, J in jobs.items():
        us = list(J); raw = llm.generate(tok, m, [c(prompts.SERVER.format(n=J[u][1], h=hist(J[u][0]))) for u in us], 200 * N, 16)
        out[k] = {u: (r, llm.parse(r, J[u][1])) for u, r in zip(us, raw)}; log("server", k, "done")
    del m; torch.cuda.empty_cache(); return out
cached("server_recs", server_recs)


def local_recs(name):
    tok, m = llm.load(LOCALS[name]); kw = dict(enable_thinking=False) if "Qwen" in name else {}
    jobs = {"local_only": {u: (H[u], N) for u in users}}
    for o in OBF:
        jobs[f"{o}_loc"] = {u: (SPLIT[o][u][1], ob.counts(*SPLIT[o][u], N)[1]) for u in users
                            if SPLIT[o][u][1] and ob.counts(*SPLIT[o][u], N)[1]}
    out = {}
    for k, J in jobs.items():
        us = list(J); raw = llm.generate(tok, m, [llm.chat(tok, prompts.LOCAL.format(n=J[u][1], h=hist(J[u][0])), **kw) for u in us], 100 * N, 48)
        out[k] = {u: (r, llm.parse(r, J[u][1])) for u, r in zip(us, raw)}; log("local", name, k, "done")
    del m; torch.cuda.empty_cache(); return out
for n in args.locals.split(","): cached(f"local_recs_{n}", lambda n=n: local_recs(n))
log("pipeline done; now run: python scripts/evaluate.py --out", args.out)
