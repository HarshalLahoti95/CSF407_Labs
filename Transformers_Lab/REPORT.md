# Transformers Lab — Report

**Course:** CS F407 — Artificial Intelligence · **Lab:** Transformers
**Companion notebook:** `transformers_lab.ipynb` (every number below is printed by it)

This lab has no separate handout PDF. Working from the three tasks it covers:
translation with an encoder–decoder model, next-token prediction with a
decoder-only model, and sentiment analysis with an encoder-only classifier. All
run on CPU with pretrained weights; nothing is trained here.

**The organising idea.** Rather than three unrelated demos, the notebook uses the
three tasks to make one architectural point: the families differ in **what
attends to what**, and that single choice determines what each can be used for.
Where possible the difference is *measured* rather than asserted.

---

## 1. Translation — MarianMT (`Helsinki-NLP/opus-mt-en-fr`)

### Why the task needs two stacks

Translation is sequence-to-sequence, and its defining awkwardness is that the
output is not an edit of the input — French word order, length and morphology all
differ, so there is no position-by-position correspondence to exploit. Hence the
split: the **encoder** reads the whole English sentence with *bidirectional*
attention; the **decoder** generates French left to right with *causal*
attention **plus cross-attention** into the encoder output. Cross-attention does
the aligning, and it is what neither other architecture here has.

### Parameter breakdown

| Component | Parameters | Share |
|---|---:|---:|
| total | 75,133,952 | 100% |
| shared embedding table | 30,471,168 | 41% |
| encoder layers only | 19,176,448 | 26% |
| decoder layers only | 25,486,336 | 34% |

The source/target embedding matrix is **shared**, so it is reachable from both
stacks; counting it in each would double-count it, which is why the table reports
the transformer layers separately.

Comparing like with like — 6 encoder layers against 6 decoder layers, embeddings
excluded — **the decoder carries 1.33× the encoder's parameters**. That ratio
*is* the cross-attention: each decoder layer has an extra attention block reading
the encoder's output on top of the self-attention the encoder layers also have.

### Translations

| English | French |
|---|---|
| The warehouse robot delivers the package to the loading bay. | Le robot d'entrepôt livre le paquet à la baie de chargement. |
| Artificial intelligence is changing how engineers write software. | L'intelligence artificielle change la façon dont les ingénieurs écrivent les logiciels. |
| The cat sat on the mat. | Le chat était assis sur le tapis. |
| I do not think this approach will work without more data. | Je ne pense pas que cette approche fonctionnera sans plus de données. |

### Sub-word tokenisation

```
"The warehouse robot delivers the package."
 → ['▁The','▁warehouse','▁robot','▁delivers','▁the','▁package','.','</s>']

"The roboticist recalibrated the electromagnetic manipulator."
 → ['▁The','▁robotic','ist','▁re','calibrat','ed','▁the',
    '▁electromagnetic','▁man','ip','ulator','.','</s>']
```

Rare words are split into pieces rather than mapped to `<UNK>`. This directly
answers a problem the **Bayesian Networks lab** hit: there, an unseen word had
probability *exactly zero* and broke the model. A sub-word vocabulary makes the
out-of-vocabulary case impossible — anything can be spelled from smaller pieces.

### Decoding strategies — and where they actually matter

| Sentence | greedy | beam 5 | agree? |
|---|---|---|:---:|
| I do not think this approach will work without more data. | Je ne pense pas que cette approche fonctionnera sans plus de données. | *identical* | **yes** |
| **I saw her duck.** | **Je l'ai vue.** | **J'ai vu son canard.** | **no** |
| The old man the boats. | Le vieux les bateaux. | Le vieil homme les bateaux. | **no** |

The result is more interesting than "beam search is better".

**On the unambiguous sentence all three strategies agree exactly.** The model is
confident enough at every step that the argmax path, the best beam and a random
draw follow the same route — so the decoding strategy is irrelevant. Most
everyday translation is like this, which is why the choice is easy to overlook.

**"I saw her duck" is where it matters.** Greedy produces *"Je l'ai vue."* — "I
saw her" — having committed early to reading *duck* as a verb and then **dropped
the word entirely**. Beam search, which scores whole sequences rather than
choosing one token at a time, recovers *"J'ai vu son canard."*

That is the precise failure mode of greedy decoding: a token that looks best
locally can leave the model in a state from which no good continuation exists,
and greedy cannot revise it. This is exactly the argument the **Agents lab** made
about reflex agents — a locally optimal choice with no lookahead can be globally
wrong — applied to decoding instead of navigation.

Sampling is the wrong tool here regardless: translation has one intended meaning,
so diversity is a cost, not a benefit.

---

## 2. Next-token prediction — GPT-2

### What GPT-2 is

A **decoder-only** transformer: causally-masked self-attention layers ending in a
projection to vocabulary logits. No encoder, no cross-attention — input and
output are the same stream. It computes exactly what the Bayesian Networks lab
estimated by counting: $P(x_t \mid x_1,\dots,x_{t-1})$.

| Component | Parameters | Share |
|---|---:|---:|
| total | 124,439,808 | 100% |
| token embeddings | 38,597,376 | 31% |
| position embeddings | 786,432 | 1% |
| transformer blocks | 85,054,464 | 68% |

12 layers, 12 heads, $d = 768$, 1024-token context, 50,257 vocabulary. **Output
head tied to the input embedding: True** — the same $V \times d$ matrix maps
tokens in and logits out, halving the cost of the model's largest tensor.

### Causal masking — verified, not assumed

Two continuations of the same 6-token prefix, differing only *after* position 6:

```
Max difference in the logits at positions 0..5 : 0.00e+00
Identical: True
```

Confirmed: position $t$'s prediction depends only on positions $\le t$. This is
what makes the model autoregressive, and why one forward pass can score every
position during training while generation must still proceed one token at a time.

### Inspecting $P(\text{next} \mid \text{context})$ directly

| Context | Entropy (nats) | Top continuations | Top-10 mass |
|---|---:|---|---:|
| "The warehouse robot delivers the" | 7.01 | goods .077, food .032, robot .021 | 0.224 |
| "The capital of France is" | 6.00 | the .085, now .048, a .046, France .032, **Paris .032** | 0.359 |
| "2 + 2 =" | 5.12 | 3 .102, 1 .092, 2 .084, 0 .075, **4 .055** | 0.525 |

(Uniform entropy over 50,257 tokens would be 10.82 nats.)

Two honest observations. GPT-2 gives *Paris* only **0.032** for the capital of
France — ranked fifth, behind function words — and ranks *4* only **fifth** for
"2 + 2 =". This is a 124M-parameter model from 2019 doing next-token statistics,
not reasoning; it is a useful corrective to treating any language model's fluency
as evidence of knowledge.

### Temperature

| $T$ | Entropy (nats) | Top-1 prob | Tokens covering 90% mass |
|---:|---:|---:|---:|
| 0.3 | 0.434 | 0.9144 | 1 |
| 0.7 | 4.165 | 0.2985 | 265 |
| 1.0 | 7.010 | 0.0772 | 3,155 |
| 1.5 | 8.976 | 0.0119 | 11,947 |
| 2.0 | 9.693 | 0.0034 | 18,759 |

Low $T$ sharpens toward the argmax; high $T$ flattens toward uniform. $T \to 0$
is greedy decoding, $T \to \infty$ is uniform sampling. **Temperature changes
nothing about the model** — the logits are fixed — it is purely a decoding-time
reshaping, which is why it can be tuned per request without retraining.

---

## 3. Sentiment analysis — DistilBERT fine-tuned on SST-2

### Why encoder-only

Sequence classification is many tokens in, one label out. Bidirectional attention
is right *and* safe here: there is no generation step, so no risk of leaking
future information — the whole input is legitimately available at once. A causal
model would have to read left to right, so its representation of the first word
could not account for the last, a real handicap when sentiment turns on a final
clause.

| Component | Parameters | Share |
|---|---:|---:|
| total | 66,955,010 | 100% |
| pretrained encoder | 66,362,880 | **99.1%** |
| classification head | 592,130 | **0.9%** |

**The head is 0.9% of the model.** Essentially all capability lives in the
pretrained encoder; fine-tuning adapted those weights and trained a tiny
classifier on top. That asymmetry is the entire economic argument for
pretraining.

### Results, including deliberately hard cases

| Text | Label | Confidence |
|---|:---:|---:|
| This laboratory was genuinely interesting and I learned a lot. | POSITIVE | 0.9989 |
| The instructions were confusing and the code did not run. | NEGATIVE | 0.9980 |
| **The film was long.** *(no sentiment word)* | NEGATIVE | **0.9965** |
| **It is not bad at all.** *(negated negative)* | POSITIVE | **0.9996** |
| **I wanted to love this, but it fell apart...** *(contrastive)* | NEGATIVE | **0.9996** |
| **A masterpiece of tedium.** *(sarcasm)* | NEGATIVE | **0.9863** |

Every row sums to 1 — the same softmax + cross-entropy pairing as the three-class
experiment in the Neural Models lab, with $K = 2$.

The model handles negation, the contrastive clause and even the sarcasm
correctly. But note **every verdict exceeds 0.98 confidence, including "The film
was long."**, which carries no sentiment word at all — there the model is
importing a prior from its training distribution (in movie reviews, "long" skews
negative) rather than reading sentiment from the sentence.

**The label set is binary by construction.** SST-2 has only POSITIVE and
NEGATIVE, so a genuinely neutral sentence *must* be forced into one of two
classes, and softmax will still report a confident probability for it. This is a
property of the label set, not a failure of the network — and it is exactly the
"output layer and loss must be chosen together" point from the Neural Models lab,
arriving from the other direction. The output space has to contain the answers
you need before the numbers mean anything.

---

## 4. The three architectures compared

| | MarianMT (enc–dec) | GPT-2 (dec-only) | DistilBERT (enc-only) |
|---|---|---|---|
| task shape | seq → seq | seq → next token | seq → label |
| encoder attention | bidirectional | — | bidirectional |
| decoder attention | causal | causal | — |
| **cross-attention** | **yes** | no | no |
| parameters | 75.1M | 124.4M | 67.0M |
| layers | 6 + 6 | 12 | 6 |
| vocabulary | 59,514 | 50,257 | 30,522 |
| output | token distribution | token distribution | 2 class logits |
| trained by | parallel-corpus seq2seq | next-token prediction | MLM, then SST-2 |

Three rules follow from the attention pattern:

1. **Need to generate?** The decoder must be causally masked — it cannot attend
   to tokens it has not produced. (Verified directly in Part 2.)
2. **Need to condition on a separate sequence?** You need cross-attention, i.e.
   an encoder–decoder. GPT-2 has no mechanism for a second sequence; prompting
   works by concatenating into *one* stream.
3. **Only need to read?** Drop the decoder — bidirectional attention is strictly
   more informative and there is no leakage risk without generation.

The modern twist is that rule 2 has been partly overturned in practice: large
decoder-only models translate well by putting both sentences in one stream,
trading cross-attention's architectural bias for scale. That is an empirical
result about capacity, not a refutation — a 75M-parameter model still benefits
from having the alignment structure built in rather than learned.

---

## 5. The same conditional, three times

| | Bayesian Networks lab | Neural Models lab | This lab (GPT-2) |
|---|---|---|---|
| Representation | explicit count table | 9-parameter network | 124M-parameter transformer |
| Context | 1–2 previous tokens | one 2-bit input | up to 1024 tokens |
| Learning | counting | gradient descent | gradient descent |
| Unseen input | probability **exactly 0** | n/a | a smooth distribution |
| Output layer | normalised counts | softmax over 3 | softmax over 50,257 |
| Logit gradient | n/a | $p - y$ | $p - y$ |

Verified numerically on a real GPT-2 forward pass:

| Check | Result |
|---|---|
| softmax sums to 1 over 50,257 tokens | $|{\sum} - 1| = 2.2\times10^{-5}$ |
| invariant to a $+1000$ logit shift | True |
| $\partial L/\partial z = p - y$ | max difference $7.45\times10^{-9}$ |

**The identity derived by hand on a 3-class toy holds exactly on a
124M-parameter model with 50,257 classes.** The mathematics did not change; only
the function computing the logits did.

**What changes is parameter sharing, not scale.** The count model stores every
context's distribution separately, so nothing learned about one context transfers
to a similar one — exactly why its second-order version overfitted and assigned
probability zero to held-out sentences. A transformer maps contexts through
shared weights, so similar contexts give similar predictions and unseen contexts
still receive sensible answers. That is generalisation, and it is what the
count-based model structurally cannot do.

---

## 6. Reflection

**The architecture is a claim about the task, not a detail.** Before this lab I
would have called all three "transformers". They are — but the choice among them
is forced by the shape of the problem, and the parameter breakdown made it
concrete: the MarianMT decoder carries 1.33× the encoder's layer parameters at
equal depth, and that excess *is* the cross-attention that does the translating.

**Verification still applies to models you did not train, and it is cheap.** I
could not retrain these or inspect their training data, but I could test
structural properties: that the softmax sums to 1 over 50,257 tokens, that it is
shift-invariant, that $\partial L/\partial z = p - y$, and that the causal mask
genuinely blocks future positions. Each would fail loudly if my mental model of
the architecture were wrong. Using someone else's model does not exempt you from
checking what kind of object it is.

**The hard part of evaluation is choosing the inputs.** The classifier looks
flawless on "this was great" and "this was terrible", and those examples teach
nothing. The informative ones separate *classifying sentiment* from *matching
sentiment vocabulary*: negation, sarcasm, contrastive clauses, and sentences with
no sentiment vocabulary at all. Same principle as the Search lab's
unreachable-goal test and the Logic lab's deliberately-broken planner — **a test
only carries information if it could have failed.** The same applies to the
decoding comparison: on an easy sentence all three strategies agreed and the
experiment showed nothing; only the ambiguous sentence was informative.

**Confidence is not accuracy.** Every sentiment verdict exceeded 0.98, including
on a sentence with no sentiment content. A binary classifier reports a
probability because its label set offers no alternative.

### Limitations

These are small models — 75M, 124M and 67M parameters — and their behaviour is
not representative of current large models. GPT-2 wanders and repeats in ways a
modern model does not, and gives *Paris* only 3% for the capital of France.
Nothing here was measured against a benchmark; the outputs are **illustrations,
not evaluations**. A real assessment of translation needs a held-out parallel
test set and a metric (BLEU, COMET); a real assessment of the classifier needs a
labelled test set and a confusion matrix, not six hand-picked sentences.

---

## Reference

A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 — the
encoder–decoder transformer and the attention patterns distinguishing the three
families used here.
S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed.
