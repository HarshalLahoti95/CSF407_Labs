# Neural Models Lab — Report

**Course:** CS F407 — Artificial Intelligence
**Lab:** Neural Models — Learning, Depth, Activations, and Output Layers
**Companion notebook:** `neural_models_lab.ipynb` (all numbers below are printed by it)

Every figure in this report is reproducible: seeds are fixed, the run is CPU-only,
and the whole notebook executes in about 100 seconds.

---

## 1. Problem specification and linear separability (Task 1)

A device has two redundant binary sensors and must warn exactly when they
disagree. Input space $\mathcal{X} = \{0,1\}^2$, output space
$\mathcal{Y} = \{0,1\}$, and the dataset is the entire input space:

| $x_1$ | $x_2$ | $y$ | meaning |
|:---:|:---:|:---:|---|
| 0 | 0 | 0 | agreement |
| 0 | 1 | 1 | disagreement |
| 1 | 0 | 1 | disagreement |
| 1 | 1 | 0 | agreement |

Because the four points exhaust $\mathcal{X}$, there is no generalisation
question here. The only question is whether the function is inside the model's
hypothesis space at all.

**Why no straight boundary works.** A linear model scores
$s = w_1x_1 + w_2x_2 + b$ and would need $s(0,1) > 0$, $s(1,0) > 0$,
$s(0,0) < 0$, $s(1,1) < 0$. Adding the first two gives $w_1 + w_2 + 2b > 0$;
adding the last two gives $w_1 + w_2 + 2b < 0$. The requirements contradict each
other, so no $(w, b)$ exists.

**Prediction recorded before coding.** A single affine layer with a sigmoid
output will not reach 4/4. Since the sigmoid is monotone it only relabels the
affine score, leaving the boundary straight, and the symmetry of the
contradiction above should drive training to the compromise $p = 0.5$ everywhere,
i.e. a loss of exactly $\ln 2 \approx 0.6931$. Stacking more affine layers should
change nothing.

### Measured result

Rather than assert non-separability, the notebook **enumerates** it. All $2^4 = 16$
labellings of the four corners were tested against a dense sweep of boundary
directions:

> Linearly separable labellings: **14 of 16**.
> The two exceptions are exactly **XOR $(0,1,1,0)$ and XNOR $(1,0,0,1)$**.

The prediction about affine models was then tested directly:

| Depth | Parameters | Final loss | Probabilities | Correct | Collapses to one affine map? |
|---:|---:|---:|---|:---:|:---:|
| 1 | 3 | 0.693147 | 0.500, 0.500, 0.500, 0.500 | 2/4 | yes |
| 3 | 105 | 0.693147 | 0.500, 0.500, 0.500, 0.500 | 2/4 | yes |
| 6 | 321 | 0.693147 | 0.500, 0.500, 0.500, 0.500 | 2/4 | yes |

The loss is $\ln 2$ to six decimal places at every depth, and multiplying each
stack's weight matrices together reproduces its predictions exactly — the
depth-6 network is *numerically* a single affine map with 321 parameters.
Prediction confirmed.

---

## 2. Model design and validation criteria (Task 2)

**Architecture:** `Linear(2,2)` → nonlinear activation → `Linear(2,1)` → one
logit. Nine parameters — the minimum that can represent XOR.

**Output/loss pairing:** the model emits a raw logit and the loss is
`BCEWithLogitsLoss` rather than `Sigmoid` + `BCELoss`. Mathematically identical,
but the fused form uses the log-sum-exp trick and does not lose precision as
$|z|$ grows — and a converged XOR model becomes very confident. Sigmoid + BCE is
the right pairing because the target is a single Bernoulli event and the logit
*is* its log-odds; the pairing makes the exponential and the logarithm cancel,
leaving $\partial L/\partial z = p - y$.

**Validation criteria, fixed before any result was seen:**

| # | Criterion | Threshold | What it rules out |
|---|---|---|---|
| C1 | Loss decreases | final < 0.05 (below the $\ln 2$ affine floor) | stuck at the symmetric compromise |
| C2 | All four labels correct | 4/4 at threshold 0.5 | loss falling without the decision changing |
| C3 | Early first-layer gradient nonzero | $\lVert\nabla_{W^{(1)}}L\rVert_2 > 0$ at step 5 | hidden layer receiving no signal |
| C4 | Hidden units differentiate | rows of $W^{(1)}$ unequal at the end | two units collapsing into one feature |

C3 is deliberately weak on its own — Section 4 shows a run with a *larger*
gradient than the baseline that nonetheless never learns XOR.

---

## 3. LLM prompt and corrections made (Task 3)

The full prompt is reproduced verbatim in the notebook. It specified the four
data rows explicitly, the 2–2–1 architecture, a configurable activation, logits
with `BCEWithLogitsLoss`, a `forward` returning the pre-activations by name, a
seeded full-batch CPU loop recording loss / gradient norm / hidden-row distance,
a gradient snapshot at an early step, three initialisation helpers, and a
finite-difference checker.

**Three corrections were made before running the generated code:**

1. **The early-gradient snapshot was taken after `optimizer.step()`.** The values
   are still present (`step()` does not clear `.grad`), so the code ran and
   printed a plausible tensor — but it was the gradient at the *pre-update*
   parameters reported as belonging to the post-update ones, and would have
   silently become `None` under `set_to_none=True`. Moved between `backward()`
   and `step()`.
2. **The finite-difference checker ran in float32.** At `eps=1e-6` the central
   difference is dominated by rounding, so the check passes or fails essentially
   at random. Changed to build a `.double()` copy of the model.
3. **Results were reported from a single seed.** Replaced with a multi-seed
   sweep, for the reason documented in Section 5.

All three bugs were in the code that *measures* the experiment, not the code that
runs it — which is the general lesson about where human verification is needed.

**What could be verified by reading, and what could not.** By reading: the
architecture and shapes, that the dataset is the four XOR rows, that the loss
matches the output parameterisation, that `zero_grad`/`backward`/`step` are
correctly ordered, and that the gradient snapshot is taken at a valid moment
(this caught bug #1). Only by running: whether the optimiser actually converges
at this learning rate, how large the gradients really are, whether a unit
saturates or dies, and whether the analytic derivation and autograd agree.

---

## 4. Learning, backpropagation, and symmetry (Task 4 A–C)

### Part A — the baseline run

2–2–1, sigmoid hidden layer, Adam(lr = 0.1), 5000 steps, seed 2.

| Input | Target | Probability | Predicted |
|---|:---:|---:|:---:|
| (0,0) | 0 | 0.000023 | 0 |
| (0,1) | 1 | 0.999970 | 1 |
| (1,0) | 1 | 0.999970 | 1 |
| (1,1) | 0 | 0.000030 | 0 |

Initial loss 0.698454 → final loss $2.805\times10^{-5}$, a reduction of about
24,900×, crossing the $\ln 2$ floor that bounds every affine model. **All four
criteria C1–C4 passed.**

The seed is a *selected* one; Section 5 reports the full distribution over 16
seeds rather than hiding it. The handout permits changing the seed as an
engineering setting — the scientific task is untouched.

**What the hidden layer did.** Plotting the four points in hidden-activation
space shows the mechanism directly: the two "disagreement" inputs, which sat on
opposite corners of the input square (distance 1.414 apart), are mapped to
essentially the same point in $h$-space. The output layer is still a straight
cut; only the space it cuts in has changed.

### Part B — backpropagation verified three independent ways

`parameter.grad` holds $\partial L/\partial\theta$ for the scalar most recently
passed to `backward()`, at the current parameter values. For the first layer,

$$\frac{\partial L}{\partial W^{(1)}} = \Big[\big(\tfrac1N(p-y)\big)W^{(2)} \odot f'(a^{(1)})\Big]^{\!\top} X.$$

Because $L = \frac1N\sum_i L_i$ and differentiation is linear, the batch gradient
is *exactly* the mean of the per-example gradients — the $\frac1N$ passes
straight through the derivative.

| Check | Method | Result |
|---|---|---|
| 4B.1 | vs. hand-derived chain rule, float64 | max error $1.7\times10^{-18}$ |
| 4B.2 | central finite differences, all 9 parameters | worst relative error $4.9\times10^{-8}$ |
| 4B.3 | batch gradient vs. mean of the 4 per-example gradients | exact match (0.0) |

A detail worth noting from 4B.3: the per-example gradient for input $(0,0)$ is
*exactly zero*, because $\partial L/\partial W^{(1)} = \delta x^\top$ and that
input is the zero vector. It contributes only to the bias gradient. A zero
per-example gradient is not a bug.

### Part C — the symmetry experiment

Four initialisations, identical in all other respects. The diagnostic is
$\lVert w_0 - w_1\rVert_2$, the distance between what the two hidden units compute.

| Init | Row gap @0 | @50 | @final | Early $\lVert\nabla_{W^{(1)}}L\rVert$ | Final loss | Correct |
|---|---:|---:|---:|---:|---:|:---:|
| random | 0.1359 | 9.7495 | 23.5700 | $3.67\times10^{-4}$ | 0.0033 | 4/4 |
| all-zero | 0.0000 | 0.0000 | **0.000000** | **0.00** | 0.6931 | 2/4 |
| tied-W1-only | 0.0000 | 4.0166 | 7.6485 | $7.68\times10^{-4}$ | 0.4780 | 3/4 |
| fully-tied | 0.0000 | 0.0000 | **0.000000** | $6.07\times10^{-3}$ | 0.4782 | 3/4 |

**This is three distinct mechanisms, not one.** The `tied-W1-only` row was run
first by mistake — I tied only $W^{(1)}$ and expected the symmetry problem, but
symmetry broke anyway. That accident is what pins down the actual requirement.

- **all-zero:** $W^{(2)} = 0$, so $\partial L/\partial h = W^{(2)\top}\delta = 0$
  and the measured $\partial L/\partial W^{(1)}$ is *exactly zero*. The hidden
  layer is not symmetric-but-learning; it is disconnected from the learning
  signal. Only $b^{(2)}$ can move, which is why the loss parks at $\ln 2$.
- **tied-W1-only:** both units compute the same $h$, but they feed the output
  through *different* outgoing weights, so their gradient rows differ on the very
  first step and the units separate immediately.
- **fully-tied** (rows, biases *and* outgoing weights equal): the gradient is
  nonzero — in fact about 16× larger than the baseline's — but its two rows are
  identical. Equal parameters plus equal updates stay equal forever; the row gap
  measured exactly 0.0 for all 400 steps. The model holds two copies of one
  feature, making it effectively a 2–1–1 network, i.e. affine.

**Conclusion:** the rule is not "identical parameters cannot learn" but
**identical *units* cannot differentiate** — and a unit is identified by
everything attached to it, outgoing weights included. Random initialisation is
what makes units distinguishable so backpropagation can assign them different
jobs.

---

## 5. The activation experiment (Task 4D)

A single run per activation turned out to measure the seed rather than the
activation: a 2–2–1 network is the *minimum* size for XOR and its loss surface
has wide plateaus. The sweep below fixes everything except the hidden activation
and repeats over 16 seeds.

### Required result table

| Hidden activation | Final loss (median) | 4/4 correct? | Early $\lVert\nabla_{W^{(1)}}L\rVert_2$ (median) |
|---|---:|:---:|---:|
| Sigmoid | $3.466\times10^{-1}$ | yes, on 7/16 seeds | 0.0022 |
| Tanh | $3.466\times10^{-1}$ | yes, on 6/16 seeds | 0.0167 |
| ReLU | $4.774\times10^{-1}$ | yes, on 4/16 seeds | 0.0061 |

At the matched seed 2, all three reach 4/4: final losses $2.81\times10^{-5}$
(sigmoid), $1.04\times10^{-5}$ (tanh), $4.31\times10^{-6}$ (ReLU).

### Interpretation

**The two orderings disagree.** Gradient magnitude runs tanh > ReLU > sigmoid;
solve rate runs sigmoid > tanh > ReLU. Measuring both is what makes this visible.

*Magnitudes.* The gradient reaching $W^{(1)}$ carries a factor $f'(a^{(1)})$.
$\tanh'(0) = 1$ and tanh is zero-centred, so it attenuates least. $\sigma'$ peaks
at 0.25 and sigmoid's output is not zero-centred, so it attenuates most — about
$7\times$ below tanh. ReLU's *active* derivative is exactly 1, the largest of the
three, yet its measured norm sits in the middle: a ReLU unit is inactive on part
of the input set and contributes exactly zero there, dragging the four-example
average down.

*Solve rates.* These are governed by *where* the derivative vanishes. Saturated
sigmoid and tanh units had minimum derivatives around $10^{-7}$–$10^{-8}$ — tiny,
but strictly positive, so a slow unit can always be pushed back. ReLU's
derivative is *exactly* zero for negative pre-activations, so a unit that goes
negative on all four inputs is excluded permanently. The notebook measures this
rather than asserting it:

| Activation | Runs with ≥1 dead unit | Dead units total | Solve rate | Solve rate among runs with no dead unit |
|---|:---:|:---:|:---:|:---:|
| sigmoid | 0/16 | 0 | 7/16 | 7/16 |
| tanh | 0/16 | 0 | 6/16 | 6/16 |
| relu | **11/16** | **15** | 4/16 | **4/5** |

ReLU's deficit is entirely accounted for by dead units: among runs where no unit
died, it solved 4 times out of 5.

**What this does not show.** Nothing here ranks activation functions in general.
Four points, two hidden units, one optimiser, one learning rate. Width 2 is
precisely the regime that punishes ReLU hardest — one dead unit is half the model
— and it is the opposite of the regime where ReLU is normally chosen. The
*engineering observation* is "at this width, on this problem, ReLU solved least
often"; the *scientific explanation* is "ReLU's derivative is exactly zero on half
its domain, which makes unit death absorbing rather than merely slow".

A larger gradient is also not a better gradient: tanh had the largest early norm
and still solved less often than sigmoid.

### The engineering fix

| Hidden units | Parameters | Solve rate | Median final loss |
|---:|---:|:---:|---:|
| 2 | 9 | 7/16 | $3.466\times10^{-1}$ |
| 4 | 17 | 15/16 | $2.026\times10^{-5}$ |
| 8 | 33 | 16/16 | $5.638\times10^{-6}$ |

The 2–2–1 network is not unable to *represent* XOR — Part A proves it can. It is
unreliable at *finding* the representation. Expressiveness and optimisability are
different properties, and only the first is settled by the argument in Task 1.

---

## 6. Three-class extension (Task 5)

Classes: 0 = both inactive `(0,0)`, 1 = disagreement `(0,1),(1,0)`, 2 = both
active `(1,1)`. Only the head changes: `Linear(2,1)` → `Linear(2,3)`, and
`BCEWithLogitsLoss` → `CrossEntropyLoss`.

### Predictions, recorded before running

1. **Final weight shape** — PyTorch stores `Linear(in,out).weight` as `(out,in)`,
   so $(3,2)$: 6 weights + 3 biases, against the binary head's 2 + 1.
2. **Logits per example** — three, so the logit tensor is $(4,3)$. Targets are
   class *indices* of shape $(4,)$, not one-hot rows.
3. **Why softmax sums to one** — $p_k = e^{z_k}/\sum_j e^{z_j}$ shares one
   denominator across all $k$, so $\sum_k p_k = 1$ by construction. It is an
   identity, not something training must learn, which is what makes softmax right
   for *mutually exclusive* classes. Three independent sigmoids would not have it.
4. **Why the logit gradient is $p - y$** — with $L = -\log p_c = -z_c + \log\sum_j e^{z_j}$,
   the first term contributes $-1$ at $k = c$ and nothing elsewhere, while the
   log-sum-exp contributes $p_k$ to every $k$. So
   $\partial L/\partial z_k = p_k - \mathbb{1}[k=c]$. The exponential in the
   softmax and the logarithm in the cross-entropy cancel — which is *why* the
   pairing is chosen, and the same cancellation seen in the binary case.

### Results — all four predictions confirmed

Output weight shape $(3,2)$; 3 logits per example; final loss
$1.341\times10^{-5}$; predicted classes `[0, 1, 1, 2]` as required.

| Input | True class | $p_0$ | $p_1$ | $p_2$ | argmax | $\sum_k p_k$ |
|---|:---:|---:|---:|---:|:---:|---:|
| (0,0) | 0 | 0.999985 | 0.000015 | 0.000000 | 0 | 0.99999996 |
| (0,1) | 1 | 0.000006 | 0.999989 | 0.000006 | 1 | 1.00000001 |
| (1,0) | 1 | 0.000005 | 0.999989 | 0.000006 | 1 | 1.00000004 |
| (1,1) | 2 | 0.000000 | 0.000017 | 0.999983 | 2 | 1.00000003 |

**The $p - y$ form, verified numerically.** Retaining the gradient on the logits
and comparing against $(p - y)/N$ gave a maximum absolute difference of
$1.8\times10^{-12}$ over all 12 logits. Reading the signs: the true class entry is
negative (push that logit up), the others positive (push them down), and each row
sums to zero, because softmax depends only on logit *differences*.

**Shift invariance and numerical stability.** Adding 0, 100 or 1000 to all three
logits left the probability vector unchanged. But the *naive* implementation does
not survive it: at $z + 1000$, $\exp(z_k)$ overflows to `inf` and `inf/inf` gives
`nan`. Subtracting the maximum logit first makes the largest exponent exactly
$e^0 = 1$, so nothing can overflow; underflow of the small terms is harmless
since they were negligible in the sum anyway. (The float32 overflow threshold for
`exp` is about 88.7.) This is the multiclass counterpart of using
`BCEWithLogitsLoss` instead of sigmoid + `BCELoss`: same function, safer
arithmetic.

---

## 7. Reflection questions

**1. Depth vs. nonlinearity.** They are not the same resource and only one buys
expressiveness. Affine-only stacks at depths 1, 3 and 6 — the deepest with 321
parameters against the working network's 9 — all converged to $p = 0.5$
everywhere and to exactly $\ln 2$, and multiplying each stack's weights together
reproduced its predictions, confirming numerically that it *is* a single affine
map. Meanwhile XOR is one of only two Boolean functions of two variables outside
the affine hypothesis space, so this is not a marginal failure but the function
being absent entirely. Two sigmoid units fixed it, and the hidden-space plot
shows how: the final layer is still a straight cut, but the points have been
moved into coordinates where a straight cut works.

**2. Evidence of a *useful* signal, not merely a nonzero gradient.** The
`fully-tied` run makes the distinction concrete: its first-layer gradient norm
was $6.1\times10^{-3}$, an order of magnitude *larger* than the successful
baseline's $3.7\times10^{-4}$, and it never learned XOR — the gradients were
identical across the two units and could only move them together. The useful-signal
evidence in the baseline is behavioural: loss fell from 0.6931 to below $10^{-4}$,
crossing the floor that bounds every affine model; all four predictions were
correct with probabilities driven to the extremes rather than hovering near 0.5;
and the row gap grew, meaning the units genuinely differentiated. Separately,
Part B established that the gradient was the *correct* derivative at all, ruling
out a buggy implementation that descends some other function.

**3. Why identical/zero initialisation prevents distinct features.** See Section
4 Part C: all-zero gives an *exactly zero* first-layer gradient (the error is
routed back through $W^{(2)} = 0$), fully-tied gives a nonzero but *identical*
gradient across units so equality is preserved forever, and tied-W1-only breaks
symmetry immediately because the units' outgoing weights differ. Identical
parameters are therefore not sufficient — identical *units* are what matter.

**4. Effect of the activation on the gradient.** See Section 5. Engineering
observation: median early gradient norm tanh (0.0167) > ReLU (0.0061) > sigmoid
(0.0022), while solve rate runs the other way. Scientific explanation:
magnitudes follow the chain-rule factor $f'$ ($\tanh'(0) = 1$ and zero-centred;
$\sigma'\le 0.25$ and not zero-centred; ReLU exactly 1 when active but exactly 0
when not, which pulls its average down), while solve rates follow *where* the
derivative vanishes — saturation is slow but recoverable, death is permanent.
The unit census measures the latter directly: 0 dead units for sigmoid and tanh,
15 across 11 of 16 ReLU runs.

**5. Why output layer and loss are chosen together.** Together they encode a
probabilistic claim about the label, and the gradient is only well-behaved when
the two halves agree. One sigmoid logit means "a Bernoulli event" and BCE is its
negative log-likelihood; softmax over $K$ logits means "exactly one of $K$" and
cross-entropy is the categorical negative log-likelihood. In both cases the
exponential and the logarithm cancel, leaving $\partial L/\partial z = p - y$ —
bounded, correctly signed, and largest exactly when the model is most wrong
(verified to $10^{-12}$ in Task 5). Mismatch them and the cancellation is lost:
sigmoid with squared error multiplies the error by $\sigma'(z)$, which vanishes
when the model is confidently wrong — the one case where a large gradient is most
needed. Three independent sigmoids over three exclusive classes would permit
probabilities summing to 2.4, discarding the exclusivity the task actually has.
The choice must also match the task's structure: genuine multi-label problems
correctly use per-class sigmoids, and softmax there would impose an exclusivity
that does not exist.

**6. LLM productivity vs. human verification.** *Productivity:* the diagnostic
scaffolding — a training loop recording per-step loss, gradient norm and row gap,
a dataclass to carry them, a finite-difference checker with fiddly index
bookkeeping, and the plotting helpers. Mechanical, precisely specifiable, and not
where the thinking is. *Verification:* the misplaced gradient snapshot (Section
3, bug #1) ran fine and printed a plausible number that was silently mislabelled;
no test would have caught it because there was nothing to compare against. The
float32 finite-difference checker is worse — it would have "passed", and a
passing gradient check is exactly the kind of evidence that stops further
investigation. Both bugs were in the instrumentation, which encodes what counts
as evidence; that is the part I have to own.

**7. Which tests survive scaling.** *Keep* — cost tracks training steps, not
model size: loss curves against a known floor (the $\ln 2$ baseline here, a
unigram-entropy baseline for a language model); per-layer gradient norms, which
is how vanishing/exploding gradients are detected at depth; activation and
pre-activation statistics, which at scale become dead-unit and attention-entropy
monitoring; repeat-over-seeds, still the only way to separate a real effect from
initialisation luck, though it shrinks to a few seeds on a small proxy; and
shape/dtype/parameter-count assertions, which catch silent-broadcast bugs for
free. *Drop or sample* — cost tracks parameter count or output size: exhaustive
finite-difference checking needs two forward passes per scalar parameter
($2\times9 = 18$ here, $2\times10^{11}$ for a mid-size model). The right response
is to shrink it rather than abandon it — run it exhaustively on a tiny version of
the same code path, then sample a few coordinates at scale, or use
`torch.autograd.gradcheck` on individual custom layers. Likewise, printing every
probability becomes printing a summary statistic, and the exhaustive enumeration
of Task 1 has no large-scale analogue at all.

---

## 8. Scaling to language models

Next-token prediction is classification over a vocabulary of $V \approx 10^5$.
**Unchanged:** one logit per class; softmax normalising by a shared denominator;
cross-entropy; the logit gradient exactly $p - y$ with $y$ one-hot on the observed
token; and the loss as an average over positions, so the gradient remains an
average of per-position gradients. The three-class experiment is that mechanism
in miniature. **Changed:** the output matrix goes from $(3,2)$ to roughly
$(10^5, d_\text{model})$ and becomes one of the largest tensors in the model,
often tied to the input embedding; the softmax denominator becomes a reduction
over $10^5$ terms per position, so the logits alone can exceed the memory of
every activation before them — hence fused cross-entropy kernels and chunked
logit computation. "One example" becomes a sequence of correlated positions
rather than an independent sample. And the hidden layer, two units here, becomes
many attention and MLP blocks doing conceptually the same job: building a
representation in which a *linear* readout suffices.

---

## 9. Responsible use of the LLM

The LLM wrote boilerplate and suggested library calls. It did not choose the
architecture, define what would count as evidence, decide that one seed was
insufficient, or catch its own misplaced gradient snapshot. The specification in
Sections 1–2 was written before any prompt was sent; criteria C1–C4 were fixed
before any result was seen; every generated block was read before being run, and
three were corrected. All numbers here are reproducible from the seeds recorded
in the notebook, and I am responsible for their interpretation.

## Reference

S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. —
agent framing, neural models, backpropagation, activation functions, and
softmax/cross-entropy output layers.
