"""Metrics, paired bootstrap and the split by target sensitivity, from the caches written by run_pipeline.py.

--server picks the server's recommendations: gpt5 (server_recs_gpt5.pkl from scripts/gpt5_server.py, the paper's main
results; outputs go to <out>/gpt5/) or open (server_recs.pkl from run_pipeline.py; outputs go to <out>/). In the framework
the server is asked for N recommendations and the device keeps its first n_ns (privrec.obfuscators.counts).
Writes results.csv (one row per configuration), paired.csv (all users and each target group), split.csv, per_user.pkl,
target_sensitive.pkl. Usage: python scripts/evaluate.py --out runs/main --server gpt5"""
import argparse, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from privrec import data, metrics as mx, obfuscators as ob

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--server", choices=["gpt5", "open"], default="gpt5")
args = ap.parse_args()
P = lambda n: os.path.join(args.out, n)
W = lambda n: os.path.join(args.out, *(["gpt5"] if args.server == "gpt5" else []), n)
os.makedirs(os.path.dirname(W("x")), exist_ok=True)
load = lambda n: pickle.load(open(P(f"{n}.pkl"), "rb"))
D = load("data"); users, H, T, M = D["users"], D["H"], D["T"], D["meta"]
L, BP, S = load("labels"), load("bert"), load("server_recs_gpt5" if args.server == "gpt5" else "server_recs")
LOC = sorted(f[len("local_recs_"):-4] for f in os.listdir(args.out) if f.startswith("local_recs_"))
Lr = {n: load(f"local_recs_{n}") for n in LOC}
MAIN = "Llama-3.2-1B" if "Llama-3.2-1B" in Lr else LOC[0]
N = 10

sens = {a: L["p_sens"][a] > 0.5 for a in L["p_sens"]}
score = {a: (0.0 if np.isnan(s) else s) for a, s in L["score"].items()}
OBF = {"cat": ob.categorical(M), "bert": ob.bert(BP)}
SPLIT = {o: {u: ob.split(H[u], f) for u in users} for o, f in OBF.items()}
NS = {o: {u: ob.counts(*SPLIT[o][u], N)[0] for u in users} for o in OBF}

g = lambda Dd, k, u: Dd.get(k, {}).get(u, ("", []))[1]
fw = lambda o, n, u: g(S, f"{o}_obf", u)[:NS[o][u]] + g(Lr[n], f"{o}_loc", u)  # the framework's merged list
BASE, LOC1, CO, CD, BO, BD = "Baseline (R)", "Only local", "Categorical obf. only (R')", "Categorical obf. + deobf. (R'')", "BERT obf. only (R')", "BERT obf. + deobf. (R'')"
nm = lambda k, n: k if n == MAIN else {LOC1: f"Only local [{n}]", CD: f"Categorical obf. + deobf. [{n}]", BD: f"BERT obf. + deobf. [{n}]"}[k]
lists = {BASE: {u: g(S, "full", u) for u in users}, CO: {u: g(S, "cat_obf", u) for u in users}, BO: {u: g(S, "bert_obf", u) for u in users}}
for n in [MAIN] + [n for n in LOC if n != MAIN]:
    lists[nm(LOC1, n)] = {u: g(Lr[n], "local_only", u) for u in users}
    lists[nm(CD, n)] = {u: fw("cat", n, u) for u in users}
    lists[nm(BD, n)] = {u: fw("bert", n, u) for u in users}

# embeddings of the catalogue (every product with metadata) and of the recommendations; each recommendation is mapped to
# its nearest catalogue product. The cache is extended with recommendations it does not have yet.
if os.path.exists(P("embeddings.pkl")): EM = load("embeddings")
else:
    cat = sorted(M); EM = dict(catalogue=cat, E=mx.embed([data.item_text(M[a]) for a in cat]).astype(np.float16), rec_emb={})
recs = sorted({x for V in lists.values() for xs in V.values() for x in xs}); new = [r for r in recs if r not in EM["rec_emb"]]
if new: EM["rec_emb"].update(zip(new, mx.embed(new))); pickle.dump(EM, open(P("embeddings.pkl"), "wb"))
E = EM["E"].astype(np.float32); cix = {a: i for i, a in enumerate(EM["catalogue"])}
Er = np.stack([EM["rec_emb"][r] for r in recs])
nn = np.concatenate([np.argmax(Er[i:i + 2048] @ E.T, 1) for i in range(0, len(Er), 2048)])
rec_item = {r: EM["catalogue"][j] for r, j in zip(recs, nn)}; rec_cat = {r: M[a].get("main_category") or "N/A" for r, a in rec_item.items()}
per = mx.per_user(lists, users, T, M, EM["rec_emb"], rec_cat, rec_item, E, cix)

hist_all = [a for u in users for a in H[u]]
sent = lambda k: (hist_all if k.startswith("Baseline") else [] if k.startswith("Only local") else
                  [a for u in users for a in SPLIT["cat" if k.startswith("Categorical") else "bert"][u][0]])
rows = [dict(scheme=k, **per[k].mean().round(4).to_dict(), **mx.leakage(sent(k), hist_all, sens, score)) for k in lists]
pd.DataFrame(rows).to_csv(W("results.csv"), index=False)

tsens = np.array([bool(sens[T[u]]) for u in users]); grp = {"all": np.ones(len(users), bool), "sensitive": tsens, "nonsensitive": ~tsens}
pd.DataFrame([dict(target=gname, scheme=k, n=int(m.sum()), **per[k][m].mean().round(4).to_dict())
              for gname, m in grp.items() if gname != "all" for k in lists]).to_csv(W("split.csv"), index=False)
pairs = list(dict.fromkeys(p for n in LOC for p in [
    (nm(BD, n), BASE), (BO, BASE), (nm(BD, n), BO), (nm(CD, n), BASE), (CO, BASE), (nm(CD, n), CO), (nm(BD, n), nm(CD, n)), (BASE, nm(LOC1, n))]))
MET = ["cat_hr", "sem_sim", "hr10", "ret_hr1000"]
pd.DataFrame([dict(target=gname, a=a, b=b, metric=m, **mx.paired_bootstrap((per[a][m] - per[b][m]).to_numpy()[msk], seed=1))
              for gname, msk in grp.items() for a, b in pairs for m in MET]).to_csv(W("paired.csv"), index=False)
pickle.dump(per, open(W("per_user.pkl"), "wb")); pickle.dump(tsens, open(W("target_sensitive.pkl"), "wb"))
print(pd.DataFrame(rows).round(3).to_string(index=False))
