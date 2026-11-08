<div align="center">

# Hybrid Obfuscation and On-Device Deobfuscation

### for privacy-preserving LLM-based product recommendation

**Tina Khezresmaeilzadeh · Jiang Zhang · Dimitrios Andreadis · Konstantinos Psounis**
<br>University of Southern California

[![arXiv](https://img.shields.io/badge/arXiv-2505.00951-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.00951)
[![Hugging Face model](https://img.shields.io/badge/Hugging%20Face-BERT%20obfuscator-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/tina-khezresmaeilzadeh/bert-sensitive-product-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-2a78d6)](LICENSE)

[Method](#how-it-works) ·
[Results](#results) ·
[Reproduce](#reproducing-the-paper) ·
[Model](https://huggingface.co/tina-khezresmaeilzadeh/bert-sensitive-product-classifier) ·
[Citation](#citation)

</div>

---

Asking a cloud LLM for product recommendations means sending it your purchase history, including the glucose
monitor, the prenatal vitamins and the eczema cream. This repository implements a simple alternative: a small
classifier on the user's device decides which purchases reveal a health condition and **never sends them**. The server
recommends from everything else, and a small on-device LLM recommends for the withheld purchases. The two lists are
merged in proportion to the two parts of the history.

In the paper the server is GPT-5, called through the OpenAI Batch API with product text only and no user identifiers.
Every other component, including the product labels used for evaluation, runs on our own hardware with open models.

<p align="center">
  <img src="figures/pipeline.png" width="100%" alt="Pipeline: the obfuscator splits the purchase history into nonsensitive products, sent to the server-side LLM recommender, and sensitive products, which stay on the device and go to the local deobfuscator; the two recommendation lists are merged.">
</p>

## Highlights

With GPT-5 as the server, on 1,295 Amazon users (95% paired-bootstrap intervals in the paper):

- **The server is worth using.** GPT-5 reaches a category hit rate of 0.723 from the full history, against 0.632 for the
  best on-device model alone (Qwen3.5-4B).
- **BERT withholding keeps the server's utility.** It keeps 75% of sensitive purchases, and 91% of strictly sensitive
  ones, on the device; the category hit rate is 0.712 (−0.011, interval [−0.030, +0.012]).
- **Deleting whole health and beauty categories does not.** It withholds 43% of the history and lowers the category hit
  rate to 0.463.
- **The on-device deobfuscator is a trade.** For users whose next purchase is sensitive it raises the category hit rate
  (+0.049) and semantic similarity (+0.018); for the other users it replaces better GPT-5 recommendations (−0.034), with
  on-device models of 1B to 4B parameters alike.
- **Cost per user** (batch size 1): the BERT obfuscator takes 0.04 s on an RTX A6000 and 0.87 s on a CPU (8 threads).

## Use it in front of any hosted LLM

The server is treated as a black box. The framework changes only which purchases go into the request, so it works
with any hosted LLM recommender, with no access to its weights and no change on the provider's side. The obfuscator
(110M parameters) and the deobfuscator (1B–4B) run on the user's machine.

```python
from privrec.client import PrivateRecommender

def server(prompt):            # any hosted LLM: wrap your provider's chat API here
    return my_provider_chat(prompt)

rec = PrivateRecommender(server, bert_dir="tina-khezresmaeilzadeh/bert-sensitive-product-classifier",
                         local_model="unsloth/Llama-3.2-1B-Instruct", device="cpu")
history = [{"title": "Blood Glucose Monitor Kit", "main_category": "Health & Personal Care",
            "features": ["For diabetes testing"]},
           {"title": "Stainless Steel Water Bottle 32oz", "main_category": "Sports & Outdoors",
            "features": ["Insulated"]}]
recommendations, n_kept_on_device = rec.recommend(history)
```

`server` receives only the products the obfuscator did not flag. The on-device model writes the recommendations for
the withheld products while the server request is in flight, and the two lists are merged.

## How it works

| Step | Where it runs | Model in the paper |
|---|---|---|
| **Obfuscator** splits the history into nonsensitive $P_{ns}$ and sensitive $P_s$ products | device | BERT-base, fine-tuned with a class-weighted focal loss |
| **Server recommender** is asked for $n$ recommendations from $P_{ns}$; the device keeps the first $n_{ns}$ | server | any hosted LLM (GPT-5 in the paper; Qwen3.5-9B in Online Resource 1) |
| **Deobfuscator** produces $n_s$ recommendations from $P_s$ | device | Llama-3.2-1B-Instruct (Qwen3.5-2B / 4B compared) |
| **Merge** concatenates both lists, $n_s = \max(1, \mathrm{round}(n\,\lvert P_s\rvert / \lvert P\rvert))$ if $P_s$ is nonempty | device | — |

The deobfuscator runs in parallel with the server call, so it only adds latency when it is slower than the server.

## Results

1,295 Amazon users, histories of up to 20 products, 10 recommendations per user, GPT-5 as the server; 95% paired-bootstrap
confidence intervals in the paper and in `results/gpt5/paired.csv`.

<p align="center">
  <img src="figures/privacy_utility.png" width="92%" alt="Privacy-utility plot: share of sensitive history entries sent to the server against category hit rate, for all users and for users whose next purchase is sensitive.">
</p>

| Configuration | Cat@10 ↑ | Sim@10 ↑ | Sensitive sent (%) ↓ | Strictly sensitive sent (%) ↓ | History sent (%) |
|---|---:|---:|---:|---:|---:|
| Cloud only (full history) | 0.723 | 0.344 | 100.0 | 100.0 | 100.0 |
| On-device only (Llama-3.2-1B) | 0.519 | 0.275 | 0.0 | 0.0 | 0.0 |
| On-device only (Qwen3.5-4B) | 0.632 | 0.331 | 0.0 | 0.0 | 0.0 |
| Category deletion | 0.463 | 0.301 | 10.2 | 11.7 | 56.5 |
| Category deletion + on-device | 0.624 | 0.323 | 10.2 | 11.7 | 56.5 |
| **BERT withholding** | **0.712** | **0.347** | **25.1** | **9.3** | **82.5** |
| BERT withholding + on-device | 0.687 | 0.338 | 25.1 | 9.3 | 82.5 |

The on-device model in the "+ on-device" rows is Llama-3.2-1B; `results/gpt5/results.csv` has every configuration and
on-device model. **Cat@10**: a recommendation's nearest catalogue product shares the target's main category ·
**Sim@10**: highest cosine similarity to the target · **Strictly sensitive**: labelled sensitive with a sensitivity score
of at least 0.8 (products that treat, monitor or manage a specific condition). Exact-item HR@10 is below 0.5% for every
configuration.

<p align="center">
  <img src="figures/by_target.png" width="92%" alt="Utility for users whose next purchase is sensitive versus nonsensitive.">
</p>

For the 144 users whose next purchase is sensitive, the on-device recommendations restore what withholding loses. For
the other 1,151 users they take places in the list that GPT-5 would have filled better, so with BERT withholding the net
effect over all users is a small loss (−0.025 category hit rate); a 4B on-device model does not change this. With
category deletion, which withholds much more, the on-device model helps everyone (+0.161).

## What it does not protect against

> [!IMPORTANT]
> The framework guarantees that the purchases it withholds never reach the server. It does **not** stop the server from
> *inferring* a condition from the purchases it does receive: harmless-looking products can be revealing in combination.
> Item-level classification cannot see such patterns. Treat this as one layer of protection, not a guarantee of
> attribute privacy.

## Repository layout

<details>
<summary>Show the file tree</summary>

```
privrec/
  data.py          Amazon Reviews 2023 -> histories, targets and catalogue
  obfuscators.py   category deletion, BERT obfuscator, split and merge counts
  llm.py           local generation with truncation handling, list parsing
  prompts.py       the four prompts of the paper (Appendix A), verbatim
  client.py        PrivateRecommender: the framework in front of any hosted LLM (black-box server)
  metrics.py       Cat@10, Sim@10, retrieval HR / MRR, distances, leakage, paired bootstrap
scripts/
  train_obfuscator.py   fine-tune the BERT obfuscator (focal loss)
  run_pipeline.py       labels -> obfuscation -> on-device recommendations, and an open server (cached stages)
  gpt5_server.py        GPT-5 as the server through the OpenAI Batch API (prepare, submit, collect, build)
  evaluate.py           all metrics, paired tests and the split by target type (--server gpt5 | open)
  make_figures.py       the figures in this README and the paper
results/gpt5/           GPT-5 answers (indexed u0000.., no user identifiers) and aggregate results of the paper
results/open_server/    aggregate results with Qwen3.5-9B as the server (Online Resource 1)
data/                   GPT-4o labels of the obfuscator's 10,012 training products
figures/
```

</details>

## Reproducing the paper

```bash
pip install -r requirements.txt

# 1. Data: download the reviews and metadata files of the ten categories from
#    https://amazon-reviews-2023.github.io/ into data/amazon2023/
#    (<Category>.jsonl and meta_<Category>.jsonl).

# 2. Obfuscator: use the released weights (--bert-dir tina-khezresmaeilzadeh/bert-sensitive-product-classifier),
#    or train them from the released labels.
python scripts/train_obfuscator.py --data data/obfuscator_training_labels.jsonl --out models/bert_obfuscator

# 3. Labels, BERT decisions, on-device recommendations and the open server
#    (GPU; about 20 GPU-hours on one RTX A6000 for 1,295 users and three deobfuscators)
python scripts/run_pipeline.py --data-dir data/amazon2023 --users users.txt \
       --bert-dir models/bert_obfuscator --out runs/main

# 4. GPT-5 as the server (needs OPENAI_API_KEY; about $42 for the three jobs at Batch API prices in Nov 2026).
#    To reuse our answers instead, copy results/gpt5/answers_*.jsonl to runs/main/gpt5/ and run only `build`.
for j in full cat bert; do
  python scripts/gpt5_server.py prepare $j --out runs/main && python scripts/gpt5_server.py submit $j --out runs/main
done
python scripts/gpt5_server.py collect full --out runs/main   # likewise cat and bert, once the batches have finished
python scripts/gpt5_server.py build --out runs/main

# 5. Metrics, paired bootstrap, split by target type
python scripts/evaluate.py --out runs/main --server gpt5      # -> runs/main/gpt5/
```

The GPT-5 answers are indexed by the position of the user in the sorted user list, so they match a run on the same
users. The open-server results in `results/open_server/` were produced with an earlier merge rule (the server was asked
for $n_{ns}$ and $n_s$ had no minimum of one); commit `2a56b68` reproduces them.

`users.txt` lists one Amazon `user_id` per line. The list of the paper's 1,295 evaluation users is available from the
authors on request; any engaged users (at least 30 reviewed products in the ten categories) can be used instead.

**Hardware used:** 2× NVIDIA RTX A6000 (48 GB), AMD Ryzen Threadripper 3970X (32 cores), 128 GB RAM. All models are loaded from the Hugging Face Hub
(`Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `meta-llama/Llama-3.2-1B-Instruct`,
`sentence-transformers/all-MiniLM-L6-v2`, `bert-base-uncased`). For Llama-3.2-1B we used `unsloth/Llama-3.2-1B-Instruct`, an ungated copy of the same weights.

## Released artefacts

| Artefact | Where |
|---|---|
| Code, prompts, aggregate results, figures | this repository |
| GPT-5 answers of the paper's run (3,875 responses, no user identifiers) | `results/gpt5/answers_*.jsonl` |
| Obfuscator training labels (10,012 products, GPT-4o, product metadata only) | `data/obfuscator_training_labels.jsonl` |
| Fine-tuned BERT obfuscator | [![Hugging Face](https://img.shields.io/badge/🤗-bert--sensitive--product--classifier-FFD21E)](https://huggingface.co/tina-khezresmaeilzadeh/bert-sensitive-product-classifier) |

The evaluation users' identifiers are not published. Labels for the products in their histories are available from
the authors on request, because the list of those products would allow the selected users to be approximately
reconstructed from the public dataset.

## Citation

```bibtex
@article{khezresmaeilzadeh2025hybrid,
  title   = {Hybrid Obfuscation and On-Device Deobfuscation for Privacy-Preserving {LLM}-Based Product Recommendation},
  author  = {Khezresmaeilzadeh, Tina and Zhang, Jiang and Andreadis, Dimitrios and Psounis, Konstantinos},
  journal = {arXiv preprint arXiv:2505.00951},
  year    = {2025}
}
```

## License

MIT (see [LICENSE](LICENSE)). The Amazon Reviews 2023 data are subject to their own terms.
