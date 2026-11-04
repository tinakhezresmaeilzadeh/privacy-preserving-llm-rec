"""Utility and privacy metrics of the paper.

Utility (per user, then averaged):
  cat_hr      category HR@10: each recommendation is mapped to its nearest catalogue product (MiniLM cosine); a hit if any
              mapped product has the target's main category (a target without one is never a hit)
  sem_sim     semantic similarity@10: maximum cosine similarity between the target and the recommendations
  ret_hr100 / ret_hr1000 / mrr
              item-level retrieval: catalogue products are scored by their maximum cosine to the recommendations and the
              target's rank is recorded (hit if rank <= K; mrr = 1 / rank)
  l1 / l2     distance of the recommended main-category distribution to the same user's baseline distribution
  hb_share    share of recommendations mapped to the health and beauty categories; hb_dev = |hb_share - baseline's|
Privacy (pooled over users): PL_b = share of sensitive history products sent to the server; PL_s weights each by its
sensitivity score."""
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
HEALTH_BEAUTY = {"Health & Personal Care", "All Beauty", "Premium Beauty"}


@torch.no_grad()
def embed(texts, device=None, batch_size=512):
    """all-MiniLM-L6-v2 sentence embeddings (mean pooling, L2-normalised)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MINILM); m = AutoModel.from_pretrained(MINILM).to(device).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        h = m(**b).last_hidden_state; w = b.attention_mask.unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize((h * w).sum(1) / w.sum(1), dim=-1).cpu().numpy())
    return np.concatenate(out)


def per_user(lists, users, T, meta, rec_emb, rec_cat, cat_emb, cat_index, baseline="Baseline (R)"):
    """DataFrame of per-user utility metrics for every configuration in `lists` ({name: {user: [recommendation]}})."""
    cats = sorted(set(rec_cat.values())); ci = {c: i for i, c in enumerate(cats)}
    def dist(xs):
        v = np.zeros(len(cats))
        for x in xs: v[ci[rec_cat[x]]] += 1
        return v / max(1, v.sum())
    tgt = {u: cat_emb[cat_index[T[u]]] for u in users}
    tc = lambda u: meta[T[u]].get("main_category") or None
    base = {u: dist(lists[baseline][u]) for u in users}
    hb = lambda xs: np.mean([rec_cat[x] in HEALTH_BEAUTY for x in xs]) if xs else 0.0
    out = {}
    for k, V in lists.items():
        rank = []
        for u in users:
            if not V[u]: rank.append(len(cat_emb)); continue
            s = (np.stack([rec_emb[x] for x in V[u]]) @ cat_emb.T).max(0); rank.append(int((s > s[cat_index[T[u]]]).sum()) + 1)
        rank = np.array(rank)
        out[k] = pd.DataFrame(dict(
            cat_hr=[float(any(rec_cat[x] == tc(u) for x in V[u])) for u in users],
            sem_sim=[max([float(rec_emb[x] @ tgt[u]) for x in V[u]], default=0.0) for u in users],
            ret_hr100=(rank <= 100).astype(float), ret_hr1000=(rank <= 1000).astype(float), mrr=1.0 / rank,
            l1=[np.abs(dist(V[u]) - base[u]).sum() for u in users], l2=[np.sqrt(((dist(V[u]) - base[u]) ** 2).sum()) for u in users],
            hb_share=[hb(V[u]) for u in users]), index=users)
    for k in out: out[k]["hb_dev"] = (out[k]["hb_share"] - out[baseline]["hb_share"]).abs()
    return out


def leakage(sent, history_products, sensitive, score):
    """PL_b, PL_s and the share of history products sent. `sent`: products sent to the server (pooled over users)."""
    allsens = [a for a in history_products if sensitive[a]]; s = [a for a in sent if sensitive[a]]
    return dict(PL_b=len(s) / len(allsens), PL_s=sum(score[a] for a in s) / sum(score[a] for a in allsens),
                sent_items=len(sent) / len(history_products))


def paired_bootstrap(x, n=1000, seed=0):
    """Mean of per-user differences x with a 95% percentile interval over resampled users."""
    rng = np.random.default_rng(seed); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(diff=float(x.mean()), lo=float(lo), hi=float(hi), significant=not lo <= 0 <= hi)
