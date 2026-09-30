# Agents Lab — Report

**Course:** CS F407 — Artificial Intelligence · **Lab:** Agents
**Companion notebook:** `agents_lab.ipynb` (every number below is printed by it)

---

## 1. Understanding the problem (Task 1)

**Environment.** A 7 × 21 grid, 147 cells of which 66 are free. `S` is at
(1, 1), `G` at (1, 19). Using the AIMA task-environment vocabulary it is fully
observable, deterministic, static, discrete, single-agent and sequential. That
combination is what licenses *offline* search: the agent can compute a complete
path before moving, and nothing will invalidate it while it deliberates.

**Goal.** Be at `G`. Formally $\text{Goal}(s) \equiv (s = \text{pos}_G)$. Note
what is *not* in the goal: any particular route, and any notion of speed. This is
why a cost measure has to be added separately to prefer shorter paths.

**Actions.** Up, Down, Left, Right, one cell each. An action is valid only if the
destination is in bounds and not `#`; invalid actions are removed from the
successor function entirely rather than penalised, which is what guarantees the
agent never crosses a shelf.

**Information the agent must maintain.** Current position, the map, the goal
position — and then three things a reflex agent does not have: a **visited set**
(without it, cycles), a **frontier** (without it, no way to back out of a dead
end), and **parent pointers** (without them it can know `G` is reachable but not
how to get there).

**Why goal-based rather than simple reflex.** Demonstrated experimentally rather
than argued — see Section 3.

---

## 2. Agent design (Task 2)

Block diagram (reproduced in full in the notebook):

```
percept → [ State ] → [ Model: grid + actions ] → [ Goal: cell G ]
                                ↓
                   [ Decision: BFS over frontier
                     + visited + parent pointers ]
                                ↓
                       action (Up/Down/Left/Right)
                                ↓
                   [ Environment: 7 × 21 warehouse ]
                                ↓
                        new percept ─── loop back
```

The **Model** box is precisely what a reflex agent lacks. Without it there is no
way to ask "what would happen if…", so all that remains is reacting to the
present cell.

| Component | Choice | Reason |
|---|---|---|
| Environment | immutable object, walls as a `frozenset` | hashable, O(1) membership |
| State | `(row, col)` | fully captures the situation; hashable |
| Actions | fixed order Up, Down, Left, Right | deterministic tie-breaking → reproducible runs |
| Search | **BFS** | all moves cost 1, so BFS is optimal with no priority queue |
| Frontier | `deque` + `popleft()` | O(1); `list.pop(0)` would be O(n) |
| Visited | `set`, marked **on enqueue** | marking on dequeue lets duplicates pile up |
| Path recovery | `parent` dict walked back from `G` | avoids storing a path per frontier entry |

---

## 3. Results: the goal-based agent vs. a simple reflex agent

Both agents were implemented and run on the **same** map.

| | Goal-based (BFS) | Simple reflex |
|---|---|---|
| Reached `G` | **yes** | **no** |
| Path length | 20 moves | — |
| Nodes expanded | 55 | — |
| Nodes generated | 120 | — |
| Max frontier | 5 | — |
| Outcome | optimal path | oscillates between (1,5) and (2,5) after 7 steps |

The goal-based agent's path, `*` marking the route:

```
#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

The reflex agent, `o` marking where it actually walked — it never leaves the
opening corridor:

```
#####################
#Soooo#............G#
#.##.o..##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

### Why the reflex agent fails — its own view of the two cells it loops between

| At cell | Distance to `G` | Legal moves | Rule picks |
|---|---:|---|---|
| (1, 5) | 14 | Down→(2,5) d=15 *further*; Left→(1,4) d=15 *further* | Down → (2,5) |
| (2, 5) | 15 | Up→(1,5) d=14 *closer*; Left→(2,4) d=16 *further*; Right→(2,6) d=14 *closer* | Up → (1,5) |

**(1,5) is a local minimum** — no legal move reduces the distance, so the rule is
forced to pick a non-improving one. At **(2,5)** two moves tie at distance 14 and
the fixed action order takes Up, straight back to (1,5). Because the rule is a
*pure function of the current cell*, arriving at (1,5) again reproduces the same
decision, and the agent is trapped in a 2-cycle.

The deeper point is not the tie-break. **No memoryless rule can fix this.**
Escaping requires knowing that (1,5) has already been tried, and "already tried"
is not a property of the current percept. Breaking the cycle needs stored state —
which is exactly what the goal-based agent's visited set and frontier supply.
That is the answer to Task 1 question 5, and it is a measurement rather than an
assertion.

---

## 4. Validating the path (not just producing one)

A path that looks right is not a validated path. Seven properties were checked
independently of the search that produced them — **all passed**:

| Check | Result |
|---|:---:|
| starts at `S` | PASS |
| ends at `G` | PASS |
| every cell is passable | PASS |
| every step is one orthogonal move | PASS |
| action list matches the cell list | PASS |
| replaying the action names through the environment reaches `G` | PASS |
| length equals the independently computed shortest distance (20) | PASS |

The last two matter most: replaying the actions tests the *answer* rather than
the algorithm, and re-deriving the shortest distance by a separate BFS confirms
optimality rather than mere feasibility.

### Adversarial tests

| Test | Path found | Moves | Expanded | Expected | Verdict |
|---|:---:|---:|---:|---|:---:|
| goal adjacent to start | yes | 1 | 1 | 1 move | PASS |
| goal walled off | **no** | — | 9 | no path | PASS |
| two routes, one shorter | yes | 10 | 18 | shortest = 10 | PASS |
| start already at goal | yes | 0 | 0 | 0 moves | PASS |

The walled-off case is the important one: it confirms the agent reports failure
by exhausting the frontier rather than looping forever. A program run only on
solvable maps has not really been tested.

---

## 5. Think About It — what if the warehouse doubled in size?

**Prediction:** BFS stays correct (complete and optimal for unit costs at any
scale), but it explores in expanding rings, so its cost should track the *area*
covered. Doubling both dimensions quadruples the area, so work should roughly
quadruple — and memory should bind before time.

**Measured** over eight warehouses of growing size at fixed obstacle density:

| Free cells | Nodes expanded | Max frontier | Path length | Time (ms) |
|---:|---:|---:|---:|---:|
| 40 | 38 | 5 | 13 | 0.07 |
| 154 | 147 | 13 | 28 | 0.24 |
| 361 | 359 | 17 | 43 | 0.57 |
| 615 | 611 | 25 | 58 | 1.16 |
| 966 | 956 | 34 | 73 | 1.65 |
| 1361 | 1346 | 37 | 88 | 2.37 |
| 1845 | 1829 | 42 | 103 | 3.81 |
| 2453 | 2439 | 40 | 118 | 4.40 |

Fitted growth: **nodes expanded ∝ (free cells)^1.01** — linear in area, exactly
as predicted. Note the contrast in the last two columns: the search cost grows
~60× across the table while the *answer* only grows ~9×. That gap is the whole
argument for an informed algorithm — A\* with a Manhattan heuristic searches
preferentially toward the goal instead of in all directions, which is the subject
of the Search lab.

Other difficulties at scale: **dynamic obstacles** (other vehicles) break the
static assumption and force re-planning; **non-uniform costs** (turning,
congestion) break BFS's optimality guarantee and require Dijkstra or A\*.

---

## 6. Prompt engineering (Task 3) and answers

The full prompt is in the notebook. Its distinguishing feature is that every
requirement restates a decision already made in Task 2 — the algorithm, the
frontier data structure, when to mark visited, the successor ordering, and what
to report. That makes the output *checkable* against a list written beforehand
rather than judged on whether it "looks right".

**1. Did the LLM generate a working program first time?** It ran and returned a
valid path first time — but "runs" and "correct" are different claims, and two of
the specified details were wrong (below).

**2. How can the prompt be improved?** The version used is already the improved
one. My first attempt merely described the problem; it returned A\* with an
unexplained heuristic and a full `path` list copied onto every frontier entry
(quadratic memory). Naming the algorithm and the data structures removed the
ambiguity.

**3. Which algorithm did the LLM choose?** BFS — because I specified it.
Unconstrained, it chose A\* with Manhattan distance.

**4. Why did it select that?** Not from analysis of the cost structure. Grid
navigation is overwhelmingly presented with A\* in training material, so A\* is
the high-probability completion. It is not *wrong* — A\* with an admissible
heuristic is also optimal — but the justification offered ("A\* is efficient for
pathfinding") was generic rather than derived from this problem's uniform unit
costs.

### Corrections made before accepting the code

1. **Visited was marked on dequeue, not enqueue.** On a cyclic grid the same cell
   then gets enqueued several times before first expansion. The path stays
   correct, but the frontier grows and the expansion count is inflated.
2. **"Expanded" was conflated with "generated".** The draft counted every cell
   *generated* and labelled it "nodes expanded". Those are different measures, and
   the BFS/A\* comparison the Search lab builds on is meaningless if they are
   confused. Separated into `n_expanded` and `n_generated`.
3. **The reflex agent had no step limit**, so on the failing case it simply hung.
   Added `max_steps` plus explicit oscillation detection.

**Both of the first two bugs were in *measurement*, not mechanism.** The path was
right every time. That is the uncomfortable part: the program looked correct,
produced correct output, and would still have reported misleading numbers.

---

## 7. Reflection

**What the LLM contributed.** The mechanical parts — dataclasses, the
path-reconstruction loop, the ASCII renderer, the plotting code. Tedious to type,
easy to specify precisely, and not where the thinking is.

**What it did not do.** Decide anything. Choosing BFS over A\* came from noticing
that every move costs 1. Marking visited on enqueue came from knowing what breaks
otherwise. Implementing a reflex agent *as a control* came from wanting question
5 answered with evidence rather than prose.

**What would survive scaling.** The validation function, because it checks the
*answer* rather than the algorithm — replaying the action list through the
environment works regardless of which search produced it. The adversarial maps
too, especially the unsolvable one, since "reports failure correctly" is the
property most likely to break and least likely to be noticed.

**What would not.** Re-deriving the full ground-truth distance map to confirm
optimality: on a warehouse large enough to matter, computing the reference answer
costs as much as the search being tested, so it would have to become a spot check
on sampled goals.

---

## Reference

S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. —
agent types (simple reflex, model-based, goal-based), task-environment
properties, and uninformed search.
