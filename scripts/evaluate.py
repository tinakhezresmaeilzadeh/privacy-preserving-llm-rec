"""Metrics, paired bootstrap and the split by target sensitivity, from the caches written by run_pipeline.py.

--server picks the server's recommendations: gpt5 (server_recs_gpt5.pkl from scripts/gpt5_server.py, the paper's main
results; outputs go to <out>/gpt5/) or open (server_recs.pkl from run_pipeline.py; outputs go to <out>/). In the framework
the server is asked for N recommendations and the device keeps its first n_ns (privrec.obfuscators.counts).
Also scores the control of Section 6.4 (the same device places filled from the on-device model's list for the full history)
and the merge rules that give the device fewer places; the rule is chosen on a random half of the users and reported on
the other half. Writes results.csv (one row per configuration), paired.csv (all users and each target group), split.csv,
merge_rules.csv, per_user.pkl, target_sensitive.pkl. Usage: python scripts/evaluate.py --out runs/main --server gpt5"""
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
if not any(p.get("categories") for p in M.values()):
    sys.exit("data.pkl has no category paths (needed for Sub@10): rerun the data stage of run_pipeline.py")
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
# control: the device's places filled from its list for the full history instead of the withheld products
CTRL = "BERT obf. + full-history on-device [{}]"
for n in LOC: lists[CTRL.format(n)] = {u: g(S, "bert_obf", u)[:NS["bert"][u]] + g(Lr[n], "local_only", u)[:N - NS["bert"][u]] for u in users}
# merge rules that give the device at most the places of the proportional rule (prefixes of its lists, no new generations)
recent = lambda u, k: bool(set(H[u][-k:]) & set(SPLIT["bert"][u][1]))  # a withheld product among the k most recent purchases
RULES = {"one slot": lambda u, m: min(m, 1)}
for k in (1, 3, 5):
    RULES[f"recent {k}"] = lambda u, m, k=k: m if recent(u, k) else 0
    RULES[f"recent {k}, one slot"] = lambda u, m, k=k: min(m, 1) if recent(u, k) else 0
RULE = "BERT obf. + deobf., {} [{}]"
for n in LOC:
    for r, f in RULES.items():
        lists[RULE.format(r, n)] = {u: g(S, "bert_obf", u)[:N - (m := f(u, N - NS["bert"][u]))] + g(Lr[n], "bert_loc", u)[:m] for u in users}

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
    (nm(BD, n), BASE), (BO, BASE), (nm(BD, n), BO), (nm(CD, n), BASE), (CO, BASE), (nm(CD, n), CO), (nm(BD, n), nm(CD, n)), (BASE, nm(LOC1, n)),
    (CTRL.format(n), BO), (nm(BD, n), CTRL.format(n))]))
MET = ["sub_hr", "cat_hr", "sem_sim", "hr10", "ret_hr1000"]
pd.DataFrame([dict(target=gname, a=a, b=b, metric=m, **mx.paired_bootstrap((per[a][m] - per[b][m]).to_numpy()[msk], seed=1))
              for gname, msk in grp.items() for a, b in pairs for m in MET]).to_csv(W("paired.csv"), index=False)
# merge rule: chosen by its Sub@10 gain over BERT withholding alone on one half of the users, reported on the other half
perm = np.random.default_rng(2026).permutation(len(users)); sel, rep = np.sort(perm[:len(users) // 2]), np.sort(perm[len(users) // 2:])
v = lambda k, m="sub_hr": per[k][m].to_numpy()
mr = []
for n in LOC:
    cand = {"proportional": nm(BD, n), **{r: RULE.format(r, n) for r in RULES}}
    gain = {r: float(np.nanmean(v(k)[sel] - v(BO)[sel])) for r, k in cand.items()}; best = max(gain, key=gain.get)
    for r, k in cand.items():
        mr.append(dict(device_model=n, rule=r, chosen=r == best, selection_half_gain=gain[r],
                       **{f"report_{m}_{a}": b for m in ["sub_hr", "cat_hr"] for a, b in mx.paired_bootstrap((v(k, m) - v(BO, m))[rep], seed=1).items()}))
pd.DataFrame(mr).to_csv(W("merge_rules.csv"), index=False)
pickle.dump(per, open(W("per_user.pkl"), "wb")); pickle.dump(tsens, open(W("target_sensitive.pkl"), "wb"))
print(pd.DataFrame(rows).round(3).to_string(index=False))
