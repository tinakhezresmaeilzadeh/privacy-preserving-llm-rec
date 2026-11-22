"""Client-side use of the framework in front of a hosted LLM recommender.

The server is a black box: `server` is any function that takes a prompt string and returns the model's text, e.g. a
wrapper around a provider's chat API. Only the nonsensitive products are written into that prompt; the obfuscator and
the deobfuscator run on the user's machine, and the deobfuscator runs while the server request is in flight."""
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from . import llm, prompts
from .data import show
from .obfuscators import bert_text, counts


class PrivateRecommender:
    def __init__(self, server, bert_dir, local_model="unsloth/Llama-3.2-1B-Instruct", threshold=0.5, n=10, device="cpu"):
        self.server, self.threshold, self.n = server, threshold, n
        self.btok = BertTokenizer.from_pretrained(bert_dir)
        self.bert = BertForSequenceClassification.from_pretrained(bert_dir).to(device).eval()
        self.ltok, self.lm = llm.load(local_model, device)
        self.kw = dict(enable_thinking=False) if "Qwen" in local_model else {}

    @torch.no_grad()
    def sensitive(self, products):
        """BERT decision per product (max_length 128, as in the paper)."""
        enc = self.btok([bert_text(p) for p in products], return_tensors="pt", truncation=True, padding="max_length",
                        max_length=128).to(self.bert.device)
        return (torch.softmax(self.bert(**enc).logits, -1)[:, 1] > self.threshold).tolist()

    def recommend(self, history):
        """history: list of dicts with `title`, `main_category` and `features` (list of strings).
        Returns (recommendations, number of products kept on the device)."""
        flags = self.sensitive(history)
        p_ns = [p for p, s in zip(history, flags) if not s]; p_s = [p for p, s in zip(history, flags) if s]
        n_ns, n_s = counts(p_ns, p_s, self.n)
        h = lambda ps: "\n".join(f"{i + 1}. {show(p)}" for i, p in enumerate(ps))
        with ThreadPoolExecutor(1) as ex:
            remote = ex.submit(self.server, prompts.SERVER.format(n=self.n, h=h(p_ns))) if n_ns else None  # n, so the request does not reveal n_s
            local = llm.generate(self.ltok, self.lm, [llm.chat(self.ltok, prompts.LOCAL.format(n=n_s, h=h(p_s)), **self.kw)],
                                 100 * n_s, 1)[0] if n_s else ""
            remote = remote.result() if remote else ""
        return llm.parse(remote, self.n)[:n_ns] + llm.parse(local, n_s), len(p_s)
