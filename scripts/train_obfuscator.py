"""Fine-tune the BERT obfuscator (Section 4.1 of the paper): bert-base-uncased, class-weighted focal loss (gamma = 2,
w_c = N / (2 N_c)), 5 epochs, lr 2e-5, weight decay 0.01, batch 16, inputs truncated to 256 tokens; 70/20/10 split
stratified by label; the epoch with the best validation F1 is kept.

Input: JSONL, one product per line with title, main_category, features (list) and label (1 = sensitive, 0 = nonsensitive).
Usage: python scripts/train_obfuscator.py --data products_labelled.jsonl --out models/bert_obfuscator"""
import argparse, json, os, sys
import numpy as np, torch
from transformers import BertTokenizer, BertForSequenceClassification
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from privrec.obfuscators import bert_text

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
rows = [json.loads(l) for l in open(args.data) if l.strip()]
y = np.array([int(r["label"]) for r in rows])
idx = {"train": [], "val": [], "test": []}
for c in (0, 1):  # stratified 70/20/10
    ii = rng.permutation(np.flatnonzero(y == c)); a, b = int(0.7 * len(ii)), int(0.9 * len(ii))
    idx["train"] += ii[:a].tolist(); idx["val"] += ii[a:b].tolist(); idx["test"] += ii[b:].tolist()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(dev)
def batches(ii, bs, shuffle):
    ii = rng.permutation(ii) if shuffle else np.array(ii)
    for i in range(0, len(ii), bs):
        b = ii[i:i + bs]
        enc = tok([bert_text({"features": [], **rows[j]}) for j in b], return_tensors="pt", truncation=True, padding=True, max_length=256)
        yield enc.to(dev), torch.tensor(y[b], device=dev)

n_c = np.bincount(y[idx["train"]], minlength=2); w = torch.tensor(len(idx["train"]) / (2 * n_c), dtype=torch.float, device=dev)
def focal(logits, t, gamma=2.0):
    logp = torch.log_softmax(logits, -1).gather(1, t[:, None]).squeeze(1)
    return (-w[t] * (1 - logp.exp()) ** gamma * logp).mean()

@torch.no_grad()
def f1(ii):
    model.eval(); tp = fp = fn = 0
    for enc, t in batches(ii, 64, False):
        p = model(**enc).logits.softmax(-1)[:, 1] > 0.5
        tp += int((p & (t == 1)).sum()); fp += int((p & (t == 0)).sum()); fn += int((~p & (t == 1)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)

opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01); best = -1
for ep in range(args.epochs):
    model.train()
    for enc, t in batches(idx["train"], 16, True):
        loss = focal(model(**enc).logits, t); opt.zero_grad(); loss.backward(); opt.step()
    v = f1(idx["val"]); print(f"epoch {ep + 1}: validation F1 {v:.3f}", flush=True)
    if v > best: best = v; model.save_pretrained(args.out); tok.save_pretrained(args.out)
model = BertForSequenceClassification.from_pretrained(args.out).to(dev)
print(f"best validation F1 {best:.3f}; test F1 {f1(idx['test']):.3f}")
