# Bayesian Networks and Autoregressive Language Models — Report

**Course:** CS F407 — Artificial Intelligence · **Lab:** Bayesian Networks
**Companion notebook:** `BN_lab.ipynb` (every number below is printed by it)

Everything is plain Python — `dict`, `Counter`, `random` — with no ML library and
no pretrained model, as the handout requires. The probabilistic structure has to
stay visible, and a neural network would hide it.

---

## Q1 — Why is the chain-rule decomposition useful for generating text?

$$P(X_1,\dots,X_6) = P(X_1)\prod_{t=2}^{6} P(X_t \mid X_1,\dots,X_{t-1})$$

**It is exact, not an approximation.** The chain rule is an identity holding for
any joint distribution in any variable order. Nothing has been assumed yet; the
modelling assumptions arrive later, when we *truncate* the conditioning context.

**It converts one intractable problem into many tractable ones.** The joint over
6 positions with vocabulary $V$ has $V^6 - 1$ free parameters — for $V = 11$,
1.7 million, from six sentences. Each factor is a distribution over just $V$
outcomes. The joint is never represented.

**It matches the shape of generation.** Sampling a joint directly is hard, but
the factorisation is sequential: sample $X_1$, condition, sample $X_2$, repeat.
Each step is a draw from a small categorical distribution and the result is an
exact sample from the joint. Generation becomes a loop — exactly how every
autoregressive model, including a modern LLM, produces text.

---

## Q2 — What independence assumption does $X_1 \to X_2 \to \cdots$ encode?

The factorisation is exact; the *network* is not. Each node's only parent is its
predecessor, so $P(X_t \mid X_1,\dots,X_{t-1}) = P(X_t \mid X_{t-1})$, i.e.

$$X_t \perp\!\!\!\perp \{X_1,\dots,X_{t-2}\} \mid X_{t-1}$$

the **first-order Markov assumption**. Parameters drop from $O(V^T)$ to $O(V^2)$.
The cost is that longer-range structure becomes *unrepresentable* — a dependency
between *cat* and a word four positions later cannot be expressed at all.

---

## Q3 — Conditional probability tables

$P(w_j \mid w_i) = C(w_i,w_j)/\sum_k C(w_i,w_k)$. Selected rows:

| Context | Distribution | Transitions observed |
|---|---|---:|
| `the` | cat 0.25, dog 0.25, mat 0.1667, park 0.1667, rug 0.1667 | 12 |
| `cat` | ran 0.3333, sat 0.6667 | 3 |
| `dog` | ran 0.3333, sat 0.6667 | 3 |
| `sat` | on 1.0000 | 4 |
| `ran` | to 1.0000 | 2 |

### Zero-probability transitions

The full first-order CPT has 121 cells ($V \times V$, $V = 11$); **17 are
non-zero (14%)**, so 86% of the table is zero.

These zeros are the model's central weakness. *cat ran* and *dog ran* both occur,
so $P(\text{ran}\mid\text{cat}) > 0$ — but a pair like *mat the* never appears, so
the model assigns probability **exactly zero**, not "small". Any sentence
containing an unseen pair gets probability 0 overall. Real n-gram models always
smooth; this lab deliberately does not, so the effect is measurable in Q12.

---

## Q4–Q7 — Inspecting the implementation

**Q4. Where are transition counts stored?** In `NGramModel.counts`, a
`defaultdict(Counter)` keyed by a **context tuple**: `counts[("the",)]["cat"]` is
$C(\text{the},\text{cat})$. A parallel `context_totals` caches the denominators.
Using a tuple key rather than a bare string is what lets one class serve any
order — which is what makes the Q11 comparison controlled.

**Q5. Where is $P(X_t \mid X_{t-1})$ computed?** In `next_distribution`, as
`count / context_totals[context]`. Nothing is precomputed, so probabilities
cannot drift out of sync with counts.

**Q6. Argmax or sampling?** Both, via a `mode` argument. The difference is
fundamental. **Greedy** is a deterministic function of the context — the same
context always gives the same word, so the distribution's shape is discarded
beyond which entry is largest. **Sampling** draws from the full distribution, so
a word with probability 0.33 is chosen about a third of the time. Only sampling
uses the model *as a probability distribution*; greedy uses it as a lookup table
of most-likely-successors.

**Q7. Unseen context?** `next_distribution` returns an empty dict and generation
stops — a deliberate, visible failure. Returning a uniform or zero-filled
distribution would let the model emit output on contexts it knows nothing about,
which is worse. Asserted in the notebook.

---

## Q8 — Testing the probability model (Part VII)

For a correct model, $\sum_v P(v \mid w) = 1$ for **every** context — a property
of the model class, independent of the data, so it is a test that can fail.

**Result: both models normalise exactly**, worst $|{\sum} - 1| < 10^{-12}$ across
all contexts.

### What a total of 0.87 would tell you — demonstrated, not described

It means the implementation is wrong: numerator and denominator disagree about
which events exist, so probability mass has gone missing. The notebook builds
**two** deliberately broken models to show the two usual causes produce two
distinct signatures:

| Context | Correct | Bug A: dropped last transition | Bug B: corpus-normalised |
|---|---:|---:|---:|
| the | 1.0000 | 1.0000 | 0.2857 |
| cat | 1.0000 | 1.0000 | 0.0714 |
| sat | 1.0000 | 1.0000 | 0.0952 |
| mat | 1.0000 | **0.0000** | 0.0476 |
| rug | 1.0000 | **0.0000** | 0.0476 |
| park | 1.0000 | **0.0000** | 0.0476 |

**Bug A** (`range(order, len(sentence) - 1)`) drops each sentence's final
transition, removing all `<END>` mass. Sums are 1.0 for mid-sentence words and
**0.0** for words occurring only sentence-finally. Loud for some contexts,
invisible for others.

**Bug B** (normalising by the corpus total rather than the context total) makes
each row sum to its *share of the corpus* — the fractional values the question
describes. **This is the more dangerous bug**: every probability is still
positive, still small, and still correctly *ordered within its row*, so argmax
predictions are unaffected and greedy generation looks perfect.

Bug B is why sampling matters as a test: a uniform rescaling of a row leaves the
argmax unchanged, so greedy decoding cannot reveal it. Only the normalisation
invariant — or sampling, which uses the magnitudes — exposes it.

### Additional invariants asserted

| Check | Result |
|---|:---:|
| all probabilities in [0, 1] | True |
| `<END>` never appears as a context | True |
| `<START>` never appears as an outcome | True |
| unseen context returns an empty result | True |
| transitions counted == corpus transitions (42 == 42) | True |

The last one catches the off-by-one directly.

---

## Q9 — Are the argmax predictions what a person would expect?

| Context | Distribution | argmax |
|---|---|---|
| `the` | cat 0.250, dog 0.250, mat 0.167, park 0.167, rug 0.167 | **cat** |
| `cat` | sat 0.667, ran 0.333 | **sat** |
| `dog` | sat 0.667, ran 0.333 | **sat** |
| `sat` | on 1.000 | **on** |
| `ran` | to 1.000 | **to** |

**Partly, and the mismatches are informative.** After *sat* the model gives *on*
with probability 1.0, and after *ran*, *to* — genuine regularities that match
English. But after *the*, the argmax is whichever of cat/dog/mat/rug/park is most
frequent, which is a fact about these six sentences, not about English.

The gap is a **data gap, not a mystery**. The model has no semantics, grammar or
world knowledge — only co-occurrence counts over 36 word tokens, so it reproduces
its training statistics including their accidents. And it is not *wrong*: it
correctly reports $P(w \mid \text{context})$ for the distribution it was given.
Judging it against English is judging it against a distribution it never saw.

---

## Q10 — Greedy vs. sampling (Part X)

Over 200 generations each:

| Mode | Distinct sentences | Most common output | Its share |
|---|---:|---|---:|
| greedy | **1** | `the cat sat on the cat sat on the cat sat on ...` (hits the length cap) | 100% |
| sampling | **53** | `the mat` | 19% |

**Sampling produces far more variation**, and the reason is structural rather
than statistical. Greedy generation is a **deterministic function** of the
starting context: from `<START>` it takes the argmax, moves to a new context,
takes the argmax again — no randomness anywhere, so the same walk is retraced
every time. It cannot produce five different sentences; it produces the same one
five times. It also **actually got trapped here**, which is worth
seeing rather than just warning about. The argmax chain is
`<START> -> the -> cat -> sat -> on -> the -> cat -> ...`: because `the` returns
`cat`, `cat` returns `sat`, `sat` returns `on` and `on` returns `the`, the walk
enters a 4-cycle and never emits `<END>`, running until the length cap. Greedy
decoding produced a degenerate, repetitive non-sentence 200 times out of 200.
This is exactly why real systems use beam search, sampling with temperature, or
repetition penalties rather than plain argmax.

Sampling draws from the full conditional, so every token with non-zero
probability is chosen in proportion to its mass. **The variation is not noise
added to the model — it is the model.** Greedy decoding discards everything about
the distribution except which entry is largest.

---

## Q11 — How does the second-order model differ?

**1. Graph structure.** The chain $X_{t-1} \to X_t$ becomes a **converging
connection** $X_{t-2} \to X_t \leftarrow X_{t-1}$; $X_t$ now has in-degree 2.

**2. CPT.** Indexed by an ordered *pair*, so the table grows from $V \times V$ to
$V^2 \times V$ — 121 to 1331 possible cells, from the same 36 training tokens.

**3. Context.** Two tokens instead of one, letting the model distinguish *the cat
sat* from *the dog sat*, which a first-order model must merge into *sat*.

**4. Data required.** Far more — the binding constraint. Rows multiplied by $V$
while the corpus stayed fixed, so observations per row fall by the same factor.

---

## Q12 — Comparison (Part XIII): more context helps, and hurts

| Measure | First-order | Second-order |
|---|---:|---:|
| contexts observed | 11 | 15 |
| possible contexts ($V^{\text{order}}$) | 11 | 121 |
| **context coverage** | **100%** | **12.4%** |
| non-zero CPT entries | 17 | 19 |
| full CPT size | 121 | 1331 |
| CPT density | 14.0% | **1.43%** |
| contexts seen exactly once | 0 | 2 |
| **training perplexity** | 1.725 | **1.292** |
| **held-out coverage** | **21/21 (100%)** | **19/21 (90%)** |
| **held-out perplexity** | **1.725** | **∞** |
| distinct sentences in 200 samples (seed 2024) | 58 | 6 |

**Why more context improves prediction.** The first-order assumption discards
information by construction. Conditioning on $(X_{t-2}, X_{t-1})$ lets the model
represent dependencies the smaller one cannot express, so the best achievable
predictor is at least as good. This is why **training perplexity falls, 1.725 →
1.292**: it fits what it saw more tightly.

**Why it becomes harder to estimate.** Contexts grow as $V^{\text{order}}$, so
rows multiplied by 11 while the corpus stayed fixed. **Context coverage collapses
from 100% to 12.4%**, CPT density from 14% to 1.4%, and two contexts were seen
exactly once — a row estimated from one observation assigns probability 1.0 to
that continuation and 0 to everything else, a confident distribution supported by
a single data point.

**The measurable consequence** is in the held-out rows: coverage *falls* from
100% to 90%, because a longer context is more likely never to have appeared. With
no smoothing, a single unseen transition drives $P(\text{sentence})$ to zero, so
**held-out perplexity is infinite**.

So the richer model **fits training data better and generalises worse** —
overfitting, visible directly in the size of the conditional probability table.
Diversity tells the same story: 58 distinct outputs versus 6 (seed 2024; the
Q10 table uses a different seed and reports 53 for the first-order model), because the
second-order model has begun *memorising* the corpus rather than generalising
from it.

This is the bias–variance trade-off in its most literal form. Markov order is the
complexity knob, $V^{\text{order}}$ is the parameter count, the corpus is fixed.
The standard responses — smoothing, backing off, interpolating between orders —
all amount to admitting a high-order CPT cannot be reliably estimated and
borrowing strength from a coarser one.

---

## Q13 — Why is Approach B preferable? (Part XV)

Approach A: *"Write a Python language model for me."* Approach B: specify
$P(X_t \mid X_{t-1})$, estimated from transition counts, with sampling-based
generation.

**Specifying intended behaviour.** "A language model" admits many readings — an
RNN, a call to a pretrained transformer, a sentence shuffler. A delegates the
modelling decision, and the most likely output is a library wrapper, defeating
the lab's purpose. B fixes *which probabilistic object* is wanted, so generated
code either implements it or is wrong.

**Understanding the representation.** I have to know the model *is* a CPT before
I can ask for one. Writing the normalisation formula into the prompt forced the
decision about what the denominator is — precisely what goes wrong in Bug B.

**Validating the implementation.** A specification makes validation possible at
all. I could check `counts[context][token]` against hand-counted transitions
because I knew what it should be. Against A there is no standard to check
against; the only question available is whether the output looks like English.

**Testing probabilistic invariants.** $\sum_v P(v\mid w) = 1$ must hold for a
correct implementation regardless of data, so it is a test that can fail. Bug B
showed the signature: positive, small, plausibly ordered — and wrong. Generated
text would not have revealed it.

**Distinguishing implementation from model.** The model is $P(X_t \mid X_{t-1})$;
the implementation is one way to compute it. Keeping them separate is what let
one class serve both orders, making the Q12 comparison genuinely about Markov
order rather than about two different programs.

**The general principle:** an LLM is reliable at translating a precise
specification into code and unreliable at inventing the specification. Approach B
keeps the part it is good at.

---

## Reflection on LLM use, with a corrected example

**What it contributed.** The `NGramModel` scaffolding — the
`defaultdict(Counter)` layout, `random.choices` with weights, the `zip`-based
transition loop, the formatting helpers. Mechanical, precisely specifiable, quick
to check.

**A piece of generated code I had to correct.** The draft's `next_distribution`
computed:

```python
total = sum(sum(c.values()) for c in self.counts.values())   # WRONG
return {tok: c / total for tok, c in self.counts[context].items()}
```

— normalising by the **corpus** total rather than the **context** total. The
output looked entirely reasonable: every value a small positive number, the
ordering within each context correct (all rows scaled by the same constant, so
even the argmax was right), and `predict` returning sensible words. The only
symptom was rows summing to their share of the corpus rather than to 1.

Two things are worth recording. **The normalisation test caught it immediately** —
the very invariant Part VII asks for, which is presumably why it asks. And
**greedy generation would never have revealed it**, because uniform rescaling
does not change the argmax; only sampling, which uses the magnitudes, misbehaves.
A model can be wrong in a way its most commonly inspected output cannot show.

I also corrected the `<START>` padding for the second-order model: the draft
prepended a single `<START>`, leaving the first real word without a full-width
context. It ran without error and silently modelled a different thing.

**The pattern:** both bugs were in the *probabilistic* content — the denominator
and the conditioning context — not in the Python. The syntax was fine throughout.
That is where verification effort belongs.

---

## Q14 — What did the Bayesian-network view add?

**1. A factorisation of the joint.** It made an impossible problem possible —
$V^6-1$ parameters replaced by $T$ conditionals over $V$ outcomes. The lab is
tractable only because the joint is never represented.

**2. An explicit statement of the dependencies, hence of the assumptions.** The
graph *is* the assumption, in a form precise enough to argue about. Without it,
"the model only looks at the previous word" is a vague description of an
implementation; with it, it is a falsifiable claim about representable
distributions.

**3. A principled generation procedure.** Ancestral sampling — draw each variable
given its parents in topological order — is a *theorem* about Bayesian networks,
not a heuristic. That is why sampling left to right yields an exact sample from
the joint, and why the `generate` loop is correct rather than merely reasonable.

**4. A way to reason about increasing context.** Second order is a *graph* change
— adding a parent — with consequences readable off the structure: the CPT gains a
dimension, parameters multiply by $V$, observations per row divide by $V$. Q12
measured exactly the degradation the graph predicts.

**5. A way to test an implementation against its specification.** The one that
did the most concrete work here. A Bayesian network's CPTs must satisfy
$\sum_v P(v\mid\text{parents}) = 1$ for every parent configuration — a property
of the *model class*, holding regardless of data, therefore assertable in code.
It caught a real bug that inspecting generated text would not have.

**6. A route from the toy to the real thing.** It separates what is *the same
idea* in a modern LLM — factorisation, conditioning, normalisation, ancestral
sampling — from what is engineering: how the conditional is represented, how far
context reaches, how parameters are estimated. A transformer is not a different
kind of object from this CPT; it is a vastly better estimator of the same
conditional distribution.

**The one change that matters most is parameter sharing**, not scale. Our CPT
stores each context's distribution separately, so nothing learned about *the cat*
transfers to *the dog* — exactly why the second-order model fell apart on
held-out data. A neural network maps contexts through a shared representation, so
similar contexts give similar predictions and unseen contexts still receive a
sensible distribution. That is generalisation, and it is what the count-based
model structurally cannot do.

---

## Deliverables checklist

| Requirement | Where |
|---|---|
| First-order implementation | `NGramModel(order=1)` |
| Second-order implementation | `NGramModel(order=2)` — same class |
| CPTs for selected contexts | Q3 and Part XII |
| Examples of generated text | Parts IX, X, XIII |
| Normalisation test results | Part VII, incl. two broken-model demos |
| Answers to Q1–Q14 | each numbered section above |
| Reflection with a corrected example | above |

---

## Takeaway

$$P(x_1,\dots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_1,\dots,x_{t-1})$$

The machinery for estimating that conditional differs enormously between a
six-sentence count table and a trillion-parameter transformer. The probabilistic
question does not.

---

## Reference

S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. —
probabilistic reasoning, Bayesian networks, and conditional independence.
