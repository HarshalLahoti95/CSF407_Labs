# CS F407 — Artificial Intelligence Labs

Six laboratory submissions. Each folder contains an executed Jupyter notebook, a
written report, and the lab handout it was built from.

| Lab | Topic | Contents |
|---|---|---|
| [Agents_Lab/](Agents_Lab/) | Goal-based agents | `agents_lab.ipynb`, `REPORT.md`, handout |
| [Search_Lab/](Search_Lab/) | Search and A\* | `search_lab.ipynb`, `REPORT.md`, handout |
| [Logic_Lab/](Logic_Lab/) | Logical planning + Prolog | `logic_lab.ipynb`, `REPORT.md`, `planner.pl`, handout |
| [BN_Lab/](BN_Lab/) | Bayesian networks & n-gram LMs | `BN_lab.ipynb`, `REPORT.md`, handout |
| [Neural_Models/](Neural_Models/) | Backprop, activations, output layers | `neural_models_lab.ipynb`, `REPORT.md`, handout |
| [Transformers_Lab/](Transformers_Lab/) | Pretrained transformer families | `transformers_lab.ipynb`, `REPORT.md` |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

All notebooks are already executed with outputs saved, and all run on CPU. Seeds
are fixed, so the numbers in each `REPORT.md` reproduce exactly. Two extras:
`Logic_Lab` shells out to SWI-Prolog (`brew install swi-prolog`), and
`Transformers_Lab` downloads about 1 GB of pretrained weights on first run.

Approximate runtimes: Agents, Search, Logic and BN are 1–2 seconds each; Neural
Models is ~100 s (multi-seed sweeps); Transformers is ~25 s once weights are
cached.

## Approach

Each lab follows the workflow the handouts ask for —
**understand → specify → generate → execute → verify** — and holds to one rule:
every claim is backed by something that runs. Where a handout asks a "Think About
It" question, the notebook answers it with a measurement rather than a paragraph.

Some of the results that came out of that:

- **Agents** — the "why not a simple reflex agent?" question is answered by
  *implementing* one and printing its view of the two cells it oscillates
  between: (1,5) is a local minimum, and the tie at (2,5) sends it straight back.
- **Search** — on the handout's map, A\* expands **exactly as many states as
  BFS**. Rather than paper over it, the notebook measures the map's degree
  profile (98% of cells are corridors, branching factor ≈ 1), then adds an
  open-plan warehouse where the same heuristic saves 51%. A heuristic only helps
  where the search has choices.
- **Logic** — a deliberately broken planner, with the precondition check removed,
  returns a *one-action* plan in which the package teleports. It runs without
  error and is shorter than the correct plan. Only independent replay exposes it.
- **BN** — two deliberately broken models show the two signatures of a
  normalisation bug, including the fractional row sums the handout's Question 8
  describes. The first-vs-second-order comparison measures held-out perplexity:
  the richer model fits training data better (1.725 → 1.292) and generalises
  worse (held-out perplexity **∞**).
- **Neural Models** — backpropagation is verified three independent ways, and the
  symmetry experiment separates three distinct mechanisms, including one I only
  found by getting the experiment wrong first.
- **Transformers** — causal masking is verified rather than assumed, and
  greedy-vs-beam decoding is shown on a sentence where they actually diverge
  ("I saw her duck" — greedy drops the word entirely).

Each report states explicitly which parts were designed by hand, what an LLM
contributed, and what had to be corrected before the generated code was accepted.
