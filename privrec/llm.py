"""Local LLM helpers: loading, chat formatting, batched greedy generation and parsing of numbered lists."""
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TRUNC = "<<TRUNCATED>>"  # appended to an output that hit the token limit


def load(name, device="cuda:0"):
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok, AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device).eval()


def chat(tok, user, **kw):
    """Single-turn chat prompt. Pass enable_thinking=False for Qwen3.5 models."""
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, **kw)


@torch.no_grad()
def generate(tok, model, prompts, max_new, batch_size):
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        g = o[:, enc.input_ids.shape[1]:]
        ended = ((g == tok.pad_token_id) | (g == tok.eos_token_id)).any(1).tolist()
        out += [t if e else t + TRUNC for t, e in zip(tok.batch_decode(g, skip_special_tokens=True), ended)]
    return out


def parse(text, n):
    """Numbered lines of a generated list, at most n; an item cut off by the token limit is dropped."""
    xs = [re.sub(r"\*\*|__", "", x).strip(" -:") for x in re.findall(r"^\s*\d+[.)]\s*(.+)$", text.replace(TRUNC, ""), re.M)]
    xs = [x for x in xs if x]
    return (xs[:-1] if TRUNC in text and len(xs) <= n else xs)[:n]
