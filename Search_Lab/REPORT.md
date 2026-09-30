# Search Lab — Report

**Course:** CS F407 — Artificial Intelligence · **Lab:** Search and A\*
**Companion notebook:** `search_lab.ipynb` (every number below is printed by it)

---

## 1. Formulating the search problem (Task 0)

| Component | Specification |
|---|---|
| **States $S$** | $\{(r,c) : \text{grid}[r][c] \ne \texttt{\#}\}$ — one per free cell; 64 in the supplied map |
| **Actions $A$** | Up, Down, Left, Right — the same four everywhere |
| **Transition $T$** | $T((r,c), a) = (r,c) + \Delta a$ if in bounds and free; **undefined otherwise** |
| **Initial state $s_0$** | the cell marked `S` = (1, 1) |
| **Goal $G$** | the single cell marked `G` = (7, 15) |
| **Cost $c$** | 1 per move — uniform |

**(a) What specifies a state?** Only `(row, col)`. Nothing else changes as the
robot moves — walls are static, it carries nothing, there is no heading or
battery. This is what keeps the state space at 64. Adding a heading would
quadruple it; adding carried packages would multiply it again.

**(b) What makes an action invalid?** Out of bounds, or the destination is `#`.
Both are handled by **omitting the action from the successor function** rather
than penalising it. Omission makes "never cross a shelf" structural; a penalty
would make it merely expensive, and a long enough detour could buy its way
through a wall.

**(c) Deterministic?** Yes — $T$ is a function, each `(state, action)` has one
known outcome. Combined with full observability, a plan computed now stays valid
when executed, so the whole path can be computed offline. A stochastic version
would need a policy over states, not a fixed action sequence.

**(d) What is a solution?** A finite action sequence from $s_0$ that is defined at
every step and ends in $G$. An **optimal** solution minimises $\sum c = n$; since
all costs are 1, optimal means fewest moves. The notebook computes the true
shortest distance independently, so optimality is *checked*, not assumed:
**40 moves**.

---

## 2. Agent design (Task 1)

| Question | Decision |
|---|---|
| State representation | `(row, col)` tuple — hashable |
| Warehouse representation | parsed once into a `frozenset` of walls plus bounds |
| Valid actions | `actions(cell)` returns only legal successors, fixed order |
| Goal recognition | equality against the single goal cell |
| Frontier contents | `(f, tie_counter, g, cell)` in a `heapq` |
| Path reconstruction | `parent` dict walked back from `G` |

Reported on termination: solution found, path, path length, **states expanded**
(kept strictly separate from states *generated*), and max frontier size.

**One design detail that matters more than it looks.** The A\* frontier entry
needs a **tie-breaking counter**. Without it, `heapq` compares the next tuple
element when two nodes share $f$ — and if that element is a `(row, col)` tuple,
the search silently becomes "lowest-row-first among equals". A\* stays optimal,
but expansion counts then depend on an accident of tuple ordering rather than on
search behaviour, which would corrupt every comparison in Tasks 5 and 6.

---

## 3. Tests (Task 3)

Each test has a **known** expected answer, so it can fail.

| Test | Found | Moves | Expected | Expanded | Verdict |
|---|:---:|---:|---|---:|:---:|
| 1. original warehouse | yes | 40 | optimum = 40 | 63 | PASS |
| 2. goal adjacent to start | yes | 1 | 1 move | 1 | PASS |
| 3. goal walled off | **no** | — | no path | 9 | PASS |
| 4. two routes available | yes | 6 | shortest = 6 | 6 | PASS |

Independent validation of the Test 1 path — all seven checks passed: starts at
`S`, ends at `G`, all cells passable, every step one orthogonal move, replaying
the *action names* through the environment reaches `G`, and the length equals the
independently computed optimum.

Test 3 matters most: it confirms the program terminates by exhausting the
frontier rather than looping. Test 4 confirms the path is not merely valid but
**shortest**.

---

## 4. Where each concept lives in the code (Task 4)

| Concept | Location |
|---|---|
| State | the `(row, col)` tuple carried as `cell` |
| Action | the name yielded by `Grid.actions`, fixed Up/Down/Left/Right order |
| Transition | `Grid.actions` — invalid successors are never generated |
| Goal test | `env.is_goal(cell)` immediately after `heappop` |
| $g(n)$ | `tentative = g + 1`, stored in `g_score` |
| $h(n)$ | `h(nxt, goal)` at push time, `h` injected as a parameter |
| $f(n)$ | `tentative + h(nxt, goal)` — the heap key |
| Frontier | `heapq` of `(f, counter, g, cell)` |
| Visited | the `expanded` set **plus** `g_score` |
| Path reconstruction | `parent` dict, walked back in `reconstruct()` |

**(a)** A binary min-heap — $O(\log n)$ push/pop with the minimum-$f$ node at the
root. **(b)** `heappop` returns smallest $f$; ties broken by insertion counter
(FIFO, reproducible); the `g > g_score[cell]` check discards stale entries.
**(c)** Only at push time, so each cell's $h$ is computed once per push.
**(d)** Yes, explicitly, as the heap key. **(e)** Two mechanisms doing different
jobs: `expanded` prevents double *expansion*; `g_score` prevents re-pushing
unless the new path is strictly cheaper — which is what permits correct
re-opening under an inconsistent heuristic. A single `visited` set would conflate
them and break Task 6.

---

## 5. A\* vs BFS (Task 5) — and an unexpected result

### On the supplied map

| Measure | BFS | A\* (Manhattan) |
|---|---:|---:|
| Solution found | yes | yes |
| Path length | 40 | 40 |
| **States expanded** | **63** | **63** |
| States generated | 127 | 127 |
| Max frontier | 3 | 3 |

**A\* saved nothing.** Reporting this as a win would be dishonest, so the
notebook diagnoses it instead. The explanation is structural and measurable: a
heuristic can only help where the search has a *choice* to make. A\* differs from
BFS solely in pop order; if almost every cell has one way forward, there is no
ordering decision to improve.

**Degree profile of the supplied map:**

| Degree | Cells | Share | Meaning |
|---:|---:|---:|---|
| 1 | 1 | 2% | dead end |
| 2 | 62 | 97% | corridor — no choice |
| 3 | 1 | 2% | T-junction |

**98% of cells are corridors or dead ends.** The handout's map is a *maze*, not
an open warehouse; its effective branching factor is ≈ 1, and the measured max
frontier was only 3. With a frontier that small there is nothing for a heuristic
to reorder, and A\* degenerates into BFS that does extra arithmetic.

### On an open-plan warehouse

To find out what the heuristic is actually worth, the notebook adds a second map
— same robot, same costs, but aisles between shelf blocks. 213 free cells, **76%
of them junctions** (degree 3 or 4) against 2% in the maze.

| Measure | BFS | A\* (Manhattan) |
|---|---:|---:|
| Path length | 30 | 30 |
| **States expanded** | **201** | **98** |
| Max frontier | 11 | 33 |

**A\* expanded 103 fewer states — a 51% reduction** — with both paths optimal.

**(a)** Both found solutions on both maps. **(b)** Equal lengths, both optimal.
**(c)** A\* expanded fewer *only on the open map*.

**(d) Why might A\* expand fewer states?** BFS orders the frontier by $g$ alone,
expanding in rings of equal distance *from the start*, growing outward in every
direction. A\* orders by $f = g + h$, so among cells of equal $g$ it prefers
those pointing at the goal. But "might" is doing real work: the heuristic's value
needs **both** a heuristic carrying real information **and** a search that faces
choices. In a corridor maze the second condition fails and no heuristic can help.
That is the honest version of the handout's own caution that *A\* is not better
simply because it is "more intelligent."*

---

## 6. Investigating the heuristic (Task 6)

### Why Manhattan is appropriate

The robot moves orthogonally at unit cost, so reaching $(x_G, y_G)$ needs at
least $|x - x_G|$ vertical and $|y - y_G|$ horizontal moves, and no move counts
for both. $h_M$ is therefore the *exact* cost of the unobstructed route — the
solution to a relaxed problem with walls deleted. That makes it **admissible**
(deleting walls can only make things easier, so $h \le h^*$ — this is what
guarantees optimality) and **consistent** (one move changes $h_M$ by exactly
$\pm 1$ at cost 1 — this is what lets A\* close a node on first expansion).

Euclidean distance is also admissible but **weaker**, since
$\sqrt{dx^2+dy^2} \le |dx|+|dy|$ — it underestimates more, so it expands at least
as many nodes.

### Results (open-plan warehouse, optimum 30)

| Heuristic | Admissible? | Path length | Optimal? | Expanded |
|---|:---:|---:|:---:|---:|
| $h = 0$ (uniform cost) | yes | 30 | yes | 208 |
| $h$ = Manhattan | yes | 30 | yes | **98** |
| $h$ = Euclidean | yes | 30 | yes | 120 |
| $h = 2 \times$ Manhattan | **no** | 30 | yes | **51** |

Exactly the predicted ordering: $h=0$ is uniform-cost search and most expensive;
Euclidean is optimal but weaker than Manhattan (120 vs 98); the inadmissible
weighted heuristic is fastest (51) but carries no guarantee — it *happened* to
return an optimal path here, which is luck, not a promise.

On the maze all four heuristics expanded 63 and returned 40 — as the branching
analysis predicts, the map cannot distinguish them.

### The weight sweep — where does the guarantee break?

Sweeping $w$ in $h_w = w \cdot h_M$:

| $w$ | Maze: length / expanded | Open plan: length / expanded |
|---:|---|---|
| 0.0 | 40 / 63 | 30 / 208 |
| 0.5 | 40 / 63 | 30 / 188 |
| 1.0 | 40 / 63 | 30 / 98 |
| 1.5 | 40 / 63 | 30 / 51 |
| 2.0 | 40 / 63 | 30 / 51 |
| 3.0 | **48** / 63 | 30 / 54 |
| 10.0 | **48** / 57 | **36** / 54 |

**Every $w \le 1$ returned an optimal path on both maps**, as the admissibility
theorem requires (asserted in code). Optimality first failed at $w = 3$ on the
maze and $w = 10$ on the open plan.

The two maps degrade differently, and instructively. On the open plan the
standard trade-off appears: raising $w$ expands far fewer states (208 → 51) until
eventually the path degrades. On the maze, the **expansion curve is flat** — with
branching ≈ 1 there is nothing to reorder, so raising $w$ buys no speed at all —
**yet the path still degrades**, 40 → 48 moves at $w \ge 3$, because an
overestimating $f$ can make A\* pop the goal via a worse route before the cheaper
one is explored.

**Flat cost, degraded answer.** On the maze an aggressive heuristic is pure loss:
it pays the optimality guarantee for a speedup that never arrives. The usual
speed-for-optimality trade is only *available* where the search has choices.

The extremes: $w = 0$ gives $f = g$, uniform-cost search — optimal, slowest;
$0 < w \le 1$ is A\* proper — optimal, faster as $w$ grows; $w \to \infty$ gives
$f \approx h$, greedy best-first — fastest, no guarantee.

---

## 7. Evaluating the LLM-generated agent (Task 7)

**1. Correct immediately:** map parsing, the Manhattan heuristic, `parent`-dict
reconstruction, the overall loop shape. Structure right, details wrong.

**2. Bugs found — three:**
1. **Goal test on node *generation* rather than on pop.**
2. **Heap entries `(f, cell)` with no tie-break counter.**
3. **Cells skipped via a `visited` set** instead of re-opened when a cheaper path
   appears.

**3. How I found them:** by reading the code against the Task 1 specification —
**not** by running it. This is the crux: **all three bugs are invisible on the
supplied map.** Manhattan is consistent there, and under a consistent heuristic
both generation-time goal testing and no-re-opening happen to give the right
answer. They only produce wrong results under the inadmissible heuristic of Task
6 — the very experiment the lab asks for. Had I accepted the code because it
"worked", I would have seen wrong path lengths in Task 6 and probably blamed the
heuristic rather than the implementation.

**4. Unfamiliar at first:** the stale-entry pattern — re-pushing a cell when a
cheaper path is found and discarding the outdated heap entry at pop time via
`if g > g_score[cell]: continue`. It looks wasteful until you realise `heapq` has
no decrease-key, so lazy deletion is the standard workaround.

**5. Modifications:** the three fixes above, plus separating `n_expanded` from
`n_generated`, which the draft conflated. Every comparison in Tasks 5–6 depends
on that distinction.

**6. Most useful tests:** Test 3 (unreachable goal), because
termination-on-failure is most likely to break and least likely to be noticed;
and the optimality check against an independently computed BFS distance map,
because it tests the *answer* rather than the algorithm. The **degree-profile
measurement** was the most *informative* — it explained a result I could not
otherwise have made sense of.

**7. Could I have trusted it untested?** No — and running it on the given map
would not have counted as testing. It produced a correct, optimal path while
containing three defects, and produced a comparison result (A\* ≈ BFS) that would
have looked like a bug had I not measured the branching factor.

**8. What I understand now that I did not before.** Two things. That
admissibility buys optimality while consistency buys expand-each-node-once, and
these are genuinely different guarantees. And — this only came from the
experiment — that **a heuristic's value depends on the topology of the search
space, not just on the heuristic.** I had assumed A\* beats BFS on grids. On the
handout's maze it does not and cannot, because 98% of cells offer no choice.
Search advice is worthless where there is nothing to decide.

---

## 8. Final reflection

**1. Why formulate before implementing?** The formulation determines which
algorithms are applicable. Noticing every move costs 1 is what makes BFS optimal
and a priority queue unnecessary; noticing $T$ is deterministic and the map fully
observable is what licenses offline planning over a policy. Skip it and the
choices get made by default — or, with an LLM, by whatever is most common in its
training data.

**2. In what sense is A\* "informed"?** It uses knowledge external to the search
graph: an estimate of cost *remaining*. BFS orders by $g$ and knows only where it
has been; A\* orders by $g + h$ and has an opinion about where it is going. That
opinion comes from a relaxed problem — here, the route cost with walls deleted.

**3. Why does the heuristic matter?** It controls correctness and cost.
Admissibility is the precondition for optimality; within it, a tighter heuristic
expands fewer nodes (Manhattan 98 vs Euclidean 120). Push past admissibility and
you buy speed with the guarantee. The experiments add a caveat theory alone does
not: the heuristic must also have somewhere to apply — on the maze every
heuristic from $h=0$ to $10h_M$ expanded ~63 states.

**4. What did the LLM contribute?** A fast first draft with the right shape, plus
boilerplate — parsing, rendering, plotting. It compressed the typing, not the
thinking.

**5. What could go wrong if an engineer accepted it untested?** Exactly what
nearly happened. The program ran, returned a valid optimal path, and would have
passed any demonstration on the supplied map — while carrying three defects that
only surface under the conditions the next experiment creates. The failure mode
is not code that breaks loudly; it is code that is right for the wrong reason and
stays quiet until the assumptions shift.

---

## Reference

S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. —
problem formulation, uninformed search, heuristic search, admissibility and
consistency.
