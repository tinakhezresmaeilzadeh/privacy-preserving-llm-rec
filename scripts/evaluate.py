"""Metrics, paired bootstrap and the split by target sensitivity, from the caches written by run_pipeline.py.

Writes to --out: results.csv (one row per configuration), paired.csv, split.csv, split_paired.csv, per_user.pkl,
target_sensitive.pkl. Usage: python scripts/evaluate.py --out runs/main"""
import argparse, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from privrec import data, metrics as mx, obfuscators as ob

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
P = lambda n: os.path.join(args.out, n)
load = lambda n: pickle.load(open(P(f"{n}.pkl"), "rb"))
D = load("data"); users, H, T, M = D["users"], D["H"], D["T"], D["meta"]
L, BP, S = load("labels"), load("bert"), load("server_recs")
LOC = sorted(f[len("local_recs_"):-4] for f in os.listdir(args.out) if f.startswith("local_recs_"))
Lr = {n: load(f"local_recs_{n}") for n in LOC}
MAIN = "Llama-3.2-1B" if "Llama-3.2-1B" in Lr else LOC[0]

sens = {a: L["p_sens"][a] > 0.5 for a in L["p_sens"]}
score = {a: (0.0 if np.isnan(s) else s) for a, s in L["score"].items()}
OBF = {"cat": ob.categorical(M), "bert": ob.bert(BP)}
SPLIT = {o: {u: ob.split(H[u], f) for u in users} for o, f in OBF.items()}

g = lambda Dd, k, u: Dd.get(k, {}).get(u, ("", []))[1]
lists = {"Baseline (R)": {u: g(S, "full", u) for u in users},
         "Only local": {u: g(Lr[MAIN], "local_only", u) for u in users},
         "Categorical obf. only (R')": {u: g(S, "cat_obf", u) for u in users},
         "Categorical obf. + deobf. (R'')": {u: g(S, "cat_srv", u) + g(Lr[MAIN], "cat_loc", u) for u in users},
         "BERT obf. only (R')": {u: g(S, "bert_obf", u) for u in users},
         "BERT obf. + deobf. (R'')": {u: g(S, "bert_srv", u) + g(Lr[MAIN], "bert_loc", u) for u in users}}
for n in LOC:
    if n == MAIN: continue
    lists[f"Only local [{n}]"] = {u: g(Lr[n], "local_only", u) for u in users}
    lists[f"Categorical obf. + deobf. [{n}]"] = {u: g(S, "cat_srv", u) + g(Lr[n], "cat_loc", u) for u in users}
    lists[f"BERT obf. + deobf. [{n}]"] = {u: g(S, "bert_srv", u) + g(Lr[n], "bert_loc", u) for u in users}

# embeddings: recommendations, and the catalogue (every product with metadata) for mapping and retrieval
if not os.path.exists(P("embeddings.pkl")):
    cat = sorted(M); recs = sorted({x for V in lists.values() for xs in V.values() for x in xs})
    E = mx.embed([data.item_text(M[a]) for a in cat]); Er = mx.embed(recs)
    nn = np.concatenate([np.argmax(Er[i:i + 2048] @ E.T, 1) for i in range(0, len(Er), 2048)])
    pickle.dump(dict(catalogue=cat, E=E.astype(np.float16), rec_emb=dict(zip(recs, Er)),
                     rec_cat={r: M[cat[j]].get("main_category") or "N/A" for r, j in zip(recs, nn)}), open(P("embeddings.pkl"), "wb"))
EM = load("embeddings"); E = EM["E"].astype(np.float32); cix = {a: i for i, a in enumerate(EM["catalogue"])}
per = mx.per_user(lists, users, T, M, EM["rec_emb"], EM["rec_cat"], E, cix)

hist_all = [a for u in users for a in H[u]]
sent = lambda k: (hist_all if k.startswith("Baseline") else [] if k.startswith("Only local") else
                  [a for u in users for a in SPLIT["cat" if k.startswith("Categorical") else "bert"][u][0]])
rows = [dict(scheme=k, **per[k].mean().round(4).to_dict(), **mx.leakage(sent(k), hist_all, sens, score)) for k in lists]
pd.DataFrame(rows).to_csv(P("results.csv"), index=False)

MET = ["cat_hr", "sem_sim", "l1", "l2", "ret_hr100", "ret_hr1000", "mrr", "hb_share", "hb_dev"]
B = "BERT obf. + deobf. (R'')"
pairs = [(B, x) for x in lists if x != B] + [(f"BERT obf. + deobf. [{n}]", B) for n in LOC if n != MAIN] + \
        [("Categorical obf. + deobf. (R'')", "Categorical obf. only (R')"), ("BERT obf. only (R')", "Categorical obf. only (R')")]
pd.DataFrame([dict(a=a, b=b, metric=m, **mx.paired_bootstrap((per[a][m] - per[b][m]).to_numpy()))
              for a, b in pairs for m in MET]).to_csv(P("paired.csv"), index=False)

tsens = np.array([bool(sens[T[u]]) for u in users]); grp = {"sensitive": tsens, "nonsensitive": ~tsens}
pd.DataFrame([dict(target=gname, scheme=k, n=int(m.sum()), **per[k][m].mean().round(4).to_dict())
              for gname, m in grp.items() for k in lists]).to_csv(P("split.csv"), index=False)
SP = [(B, "BERT obf. only (R')"), ("Categorical obf. + deobf. (R'')", "Categorical obf. only (R')"), (B, "Baseline (R)"),
      ("BERT obf. only (R')", "Baseline (R)"), ("Categorical obf. + deobf. (R'')", "Baseline (R)"), ("Categorical obf. only (R')", "Baseline (R)")] + \
     [(f"BERT obf. + deobf. [{n}]", "BERT obf. only (R')") for n in LOC if n != MAIN]
pd.DataFrame([dict(target=gname, a=a, b=b, metric=m, **mx.paired_bootstrap((per[a][m] - per[b][m]).to_numpy()[msk], seed=1))
              for gname, msk in grp.items() for a, b in SP for m in ["cat_hr", "sem_sim", "ret_hr1000", "mrr"]]).to_csv(P("split_paired.csv"), index=False)
pickle.dump(per, open(P("per_user.pkl"), "wb")); pickle.dump(tsens, open(P("target_sensitive.pkl"), "wb"))
print(pd.DataFrame(rows).round(3).to_string(index=False))
