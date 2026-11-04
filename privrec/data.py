"""Amazon Reviews 2023 -> purchase histories.

A user's reviewed products, in time order, are their purchase history (first review per product). The target is the most
recent product, the history the (up to) 20 products before it with metadata; users need at least 15 history products.
Engaged users have at least 30 reviewed products in the ten categories. The retrieval catalogue is every product in the
last 21 products of an engaged user.

Expects the official files from https://amazon-reviews-2023.github.io/ in one directory:
  <Category>.jsonl (reviews) and meta_<Category>.jsonl (product metadata) for each category in CATEGORIES."""
import json, os
from collections import defaultdict

CATEGORIES = ["Amazon_Fashion", "Beauty_and_Personal_Care", "Electronics", "Health_and_Household", "Magazine_Subscriptions",
              "Books", "Baby_Products", "Grocery_and_Gourmet_Food", "Health_and_Personal_Care", "Musical_Instruments"]
SENSITIVE_CATEGORIES = {"Health_and_Household", "Health_and_Personal_Care", "Beauty_and_Personal_Care"}  # categorical obfuscator
HISTORY, MIN_HISTORY, MIN_ENGAGED = 20, 15, 30


def _jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_sequences(data_dir, users=None):
    """{user_id: [parent_asin, ...]} in time order, first review per product only. `users` restricts the scan."""
    first = defaultdict(dict)  # user -> asin -> earliest timestamp
    for c in CATEGORIES:
        for r in _jsonl(os.path.join(data_dir, f"{c}.jsonl")):
            u = r["user_id"]
            if users is not None and u not in users: continue
            a, t = r["parent_asin"], r["timestamp"]
            if a not in first[u] or t < first[u][a]: first[u][a] = t
    return {u: [a for a, _ in sorted(d.items(), key=lambda x: (x[1], x[0]))] for u, d in first.items()}


def load_meta(data_dir, asins):
    """{parent_asin: {title, main_category, features, src}} for the requested products (src = category file)."""
    asins, meta = set(asins), {}
    for c in CATEGORIES:
        for p in _jsonl(os.path.join(data_dir, f"meta_{c}.jsonl")):
            a = p.get("parent_asin")
            if a in asins and a not in meta:
                meta[a] = {"title": p.get("title") or "", "main_category": p.get("main_category") or "",
                           "features": [str(x) for x in (p.get("features") or [])][:5], "src": c}
    return meta


def build(data_dir, user_ids):
    """Histories and targets for the given evaluation users, plus the retrieval catalogue's metadata.
    Returns dict(users, H, T, meta), with meta covering histories, targets and the catalogue."""
    seq_all = load_sequences(data_dir)
    engaged = {u: s for u, s in seq_all.items() if len(s) >= MIN_ENGAGED}
    need = {a for s in engaged.values() for a in s[-(HISTORY + 1):]}
    need |= {a for u in user_ids if u in seq_all for a in seq_all[u][-(HISTORY + 1):]}
    meta = load_meta(data_dir, need)
    H, T = {}, {}
    for u in user_ids:
        s = seq_all.get(u)
        if not s or s[-1] not in meta: continue
        h = [a for a in s[-(HISTORY + 1):-1] if a in meta]
        if len(h) >= MIN_HISTORY: H[u], T[u] = h, s[-1]
    users = sorted(H)
    return dict(users=users, H=H, T=T, meta=meta)


def features(p, k=3):
    return [str(x)[:120] for x in (p.get("features") or [])[:k]]


def show(p):
    """One history line as shown to the LLMs: [main category] title; up to three features."""
    s = f"[{p.get('main_category') or 'N/A'}] {p['title'][:150]}"
    return s + "".join(f"; {x}" for x in features(p))


def item_text(p):
    """Text embedded for retrieval and semantic similarity."""
    return f"{p['title']}. {p.get('main_category') or ''}. " + " ".join(features(p))
