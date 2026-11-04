"""Obfuscators split a history into (nonsensitive, sensitive) products; the merge rule sets how many recommendations
each side produces."""
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from .data import SENSITIVE_CATEGORIES


def categorical(meta):
    """Every product from a health or beauty category file is sensitive."""
    return lambda a: meta[a]["src"] in SENSITIVE_CATEGORIES


def bert_text(p):
    """Input format of the fine-tuned BERT obfuscator (title, main category, features)."""
    s = f"{p['title']}**\n- Main Category: {p['main_category'] or 'N/A'}\n"
    s += ("- Features:\n" + "".join(f"  - {f}\n" for f in p["features"])) if p["features"] else "- Features: N/A\n"
    return s + "\n"


@torch.no_grad()
def bert_probs(model_dir, meta, asins, device="cuda", batch_size=256):
    """P(sensitive) for each product; max_length 128 at inference."""
    tok = BertTokenizer.from_pretrained(model_dir); m = BertForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out = {}
    for i in range(0, len(asins), batch_size):
        b = asins[i:i + batch_size]
        enc = tok([bert_text(meta[a]) for a in b], return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
        out.update(zip(b, torch.softmax(m(**enc).logits, -1)[:, 1].tolist()))
    return out


def bert(probs, threshold=0.5):
    return lambda a: probs[a] > threshold


def split(history, is_sensitive):
    """(P_ns, P_s): history products kept for the server and kept on the device."""
    return [a for a in history if not is_sensitive(a)], [a for a in history if is_sensitive(a)]


def counts(p_ns, p_s, n=10):
    """(n_ns, n_s): recommendations from the server and from the device, proportional to the two parts."""
    n_s = int(round(n * len(p_s) / max(1, len(p_ns) + len(p_s)))) if p_ns else n
    return n - n_s, n_s
