# Logic Lab — Report

**Course:** CS F407 — Artificial Intelligence · **Lab:** Logical Reasoning for Planning
**Companion notebook:** `logic_lab.ipynb` · **Prolog knowledge base:** `planner.pl`

The lab's claim is **Logic + Search = Planning**. Logic decides what is
*possible* ($S \models \text{Pre}(a)$); search decides what to *try*.

---

## 1. The planning problem (Task 0)

$I = \{\text{At}(Robot,A),\ \text{At}(Package,A)\}$ and
$G = \{\text{At}(Package,C)\}$.

The goal is a **partial** state — it constrains only where the package is, not
where the robot ends up. This is why the goal test must be $G \subseteq S$ rather
than $G = S$, a point that turned out to matter (Section 3).

| Action | Preconditions | Add | Delete |
|---|---|---|---|
| $\text{Move}(x,y)$ | At(Robot,$x$) | At(Robot,$y$) | At(Robot,$x$) |
| $\text{PickUp}(P,\ell)$ | At(Robot,$\ell$) ∧ At($P$,$\ell$) | Holding($P$) | At($P$,$\ell$) |
| $\text{Drop}(P,\ell)$ | At(Robot,$\ell$) ∧ Holding($P$) | At($P$,$\ell$) | Holding($P$) |

Move exists only for connected pairs A–B, B–A, B–C, C–B. **There is no direct
A–C link** — the fact the Prolog verifier checks in Section 6.

### Which actions are applicable in $I$?

**PickUp(Package, A): yes.** Its preconditions At(Robot,A) and At(Package,A) are
both in $I$, so $I \models \text{Pre}$.

**Drop(Package, C): no.** It requires At(Robot,C) and Holding(Package); the robot
is at A and holds nothing. It appears in the action *list*, but being listed is
not the same as being applicable — and that distinction is exactly where logical
reasoning enters planning.

---

## 2. The hand-built plan (Task 1)

| State | Facts | Justification |
|---|---|---|
| $S_0$ | At(Robot,A), At(Package,A) | the given $I$ |
| $S_1$ | At(Robot,A), Holding(P) | PickUp(P,A): both preconditions hold; deletes At(P,A) |
| $S_2$ | At(Robot,B), Holding(P) | Move(A,B) |
| $S_3$ | At(Robot,C), Holding(P) | Move(B,C) |
| $S_4$ | At(Robot,C), At(Package,C) | Drop(P,C): At(Robot,C) and Holding(P) both hold |

$S_4 \supseteq G$ — **4 actions**, and that is the minimum: any valid plan needs a
PickUp and a Drop (nothing else moves the package), and travelling A→C needs at
least two Moves since there is no A–C link.

### The handout's own example sequence is invalid

The handout illustrates with Move(A,B), **PickUp(P,B)**, Move(B,C), Drop(P,C).
Replayed against the action definitions, it fails:

```
(initial)                -> {At(Package,A), At(Robot,A)}
Move(A,B)                -> {At(Package,A), At(Robot,B)}
PickUp(Package,B)        -> BLOCKED: missing ['At(Package,B)']
```

After Move(A,B) the package has not moved, so PickUp(P,B) fails its
precondition. The sequence has the right shape, the right length, and the right
actions in a sensible order — and is still wrong. This is Reflection Q3 in
miniature, and the notebook checks it rather than taking my word for it.

---

## 3. The planner, and the three bugs I corrected (Task 2)

The full prompt is in the notebook. It specified states as `frozenset`s, an
`Action` dataclass with `applicable` and `apply`, BFS over states, a subset goal
test, and a **separate** `verify_plan` that replays a plan without using the
planner.

### Corrections made before accepting the generated code

1. **The goal test was `state == goal`.** Unsatisfiable here: the final state is
   {At(Robot,C), At(Package,C)}, a proper superset of $G$. The planner reported
   "no plan found" for a problem that plainly has one.
2. **The visited set stored *plans*, not states.** Every distinct action sequence
   is a distinct plan, so nothing was ever pruned and the search explored an
   exponentially growing tree. It still terminated here because the space is
   tiny, but the expansion count was inflated and it would not scale.
3. **`apply` added before deleting.** No warehouse action has overlapping add and
   delete lists, so this changed *no output at all*.

**These three failed in three different ways, which is the lesson.** Bug 1 failed
**loudly** — impossible to miss. Bug 2 failed **quietly** — correct answers,
wrong numbers, bad scaling. Bug 3 failed **invisibly** — nothing observable, a
latent defect waiting for the domain to grow an action like "recharge while
staying put". Only the first would have been caught by running the program.

### Result (Test A)

```
Plan of length 4, found after expanding 6 states:
(initial)              {At(Package,A), At(Robot,A)}
1. PickUp(Package,A)   {At(Robot,A), Holding(Package)}
2. Move(A,B)           {At(Robot,B), Holding(Package)}
3. Move(B,C)           {At(Robot,C), Holding(Package)}
4. Drop(Package,C)     {At(Package,C), At(Robot,C)}
```

Independent replay: every action applicable = **True**, goal achieved = **True**,
length matches the hand-derived minimum of 4.

---

## 4. Testing (Task 3)

| Test | Plan found | Plan | Valid? |
|---|:---:|---|:---:|
| A: original | yes | PickUp(P,A) → Move(A,B) → Move(B,C) → Drop(P,C) | yes |
| B: PickUp removed | **no** | — | n/a |
| B2: B–C link removed | **no** | — | n/a |
| C: with an Idle action | yes | same 4-action plan | yes |
| C2: **broken planner** | yes | **Drop(Package,C)** | **NO** |

**Test B** — the planner exhausted a finite state space and stopped. It did not
hallucinate an action and did not hang.

**Test C** — with an irrelevant `Idle` action available, BFS still returns the
4-action plan; a useless action is simply never part of a shortest one. And a
state with the **robot** at C but the package at A does **not** satisfy the goal:
At(Robot,C) and At(Package,C) are different propositions, so the confusion the
handout warns about cannot arise.

### Test C2 — exhibiting the failure rather than describing it

I built a planner that skips the precondition check and ran it on the same
problem. It returned:

```
Plan: [Drop(Package,C)]

Drop(Package,C) -> BLOCKED, preconditions not satisfied:
                   ['At(Robot,C)', 'Holding(Package)']
```

**A one-action plan.** The robot drops a package it never picked up, in a room it
never travelled to, and At(Package,C) appears from nothing. The package
teleports.

Three things about this matter. It **ran fine** — no exception, no hang. Its
output is **plausibly formatted** — a list of action names, like any other plan.
And it is **shorter than the correct plan**, because removing the precondition
filter enlarges the successor set, so spurious shortcuts appear and BFS finds
them first. A shorter plan looks like a better plan. Nothing in the output
signals fiction; only independent replay does.

---

## 5. Logic and search (Task 4)

The handout's diagram, with the blank filled:

```
Current state S
      ↓
Check action preconditions        ← LOGIC
      ↓  (does S ⊨ Pre(a)?)
Apply the effects                 ← LOGIC  [the missing box]
      ↓  S' = (S \ Del) ∪ Add
Generate successor state S'
      ↓
Search over alternatives          ← SEARCH
      ↓  (BFS, skipping visited states)
Goal?  G ⊆ S'
```

Logic does **two** distinct jobs: it decides *whether* an action may be taken
(precondition entailment) and *what the world becomes* (the delete-then-add
update — the ordering is part of the semantics, not an implementation detail).
Search does neither; it only chooses which logically-permitted successor to
explore next. The interface is deliberately narrow: logic exposes a successor
function, search consumes it. Swap the domain and the same BFS works unchanged.

**The state space is finite**, which is what makes "no plan found" a *proof*. The
robot is at one of 3 locations; the package is at one of 3 or held — at most
$3 \times 4 = 12$ states. The notebook enumerates them and finds **exactly 12**.
So when the frontier empties, no plan exists; Test B's result is a result, not a
timeout.

---

## 6. Prolog as an independent verifier (Tasks 6–8)

Run against real SWI-Prolog (`/opt/homebrew/bin/swipl`), each query in a fresh
process — genuinely a separate reasoner, not a Python function in disguise.

### Task 6

| Query | Result |
|---|:---:|
| `can_move(a,b)` | **true** |
| `can_move(a,c)` | **false** |
| `can_move(b,c)` | true |
| `can_move(c,b)` | true |

**(a)** `can_move(a,b)` succeeds because `can_move(X,Y) :- connected(X,Y)`
unifies X=a, Y=b, and `connected(a,b)` is a fact — one rule application plus a
fact lookup.

**(b)** `can_move(a,c)` fails because `connected(a,c)` is not a fact and no rule
derives it. Under the closed-world assumption, what cannot be proved is taken to
be false. This is *failure-to-prove*, not a proof of falsity — a distinction that
matters whenever a knowledge base is incomplete rather than merely finite.

**(c)** The clause `can_move(X,Y) :- connected(X,Y).` **is** the implication
$\text{Connected}(X,Y) \to \text{CanMove}(X,Y)$, written as a definite (Horn)
clause with the conclusion first. Prolog uses it in one direction only — backward,
goal to subgoal — so it is an inference *rule*, not a two-way equivalence.

### Task 7 — checking Python's plan with Prolog

| Proposed step | `valid_move`? | Verdict |
|---|:---:|---|
| Move(a,b) | true | supported by the KB |
| Move(b,c) | true | supported by the KB |
| Move(a,c) *(hypothetical)* | **false** | no direct A–C connection |

Every movement in the generated plan is independently supported.

**The challenge case is the informative one.** For the hypothetical Move(a,c):

- `valid_move(a,c)` → **false** — not a legal single action
- `reachable(a,c)` → **true** — but a *is* connected to c in two steps, via b

Together these say: a plan containing Move(a,c) is invalid, yet the underlying
task is achievable. A verifier reporting only "false" would conflate *this step
is illegal* with *this goal is impossible*. Separating the one-step rule from its
transitive closure keeps those distinct.

### Task 8 — chained inference

`?- reduce_speed.` → **true**. Prolog works backwards: `reduce_speed` matches
`reduce_speed :- slippery`, so it must prove `slippery`; that matches
`slippery :- wet_road`, so it must prove `wet_road`; that is a fact, and the
chain unwinds.

$$\text{WetRoad} \;\Rightarrow\; \text{WetRoad}\to\text{Slippery} \;\Rightarrow\; \text{Slippery}\to\text{ReduceSpeed} \;\Rightarrow\; \text{ReduceSpeed}$$

Fact ⇒ Rule ⇒ Rule ⇒ Conclusion, by modus ponens twice. The programmer supplies
facts and rules; Prolog supplies the inference.

### Prolog reflection

1. **Fact vs. rule.** A fact is an unconditional assertion (empty body), true in
   every model. A rule is conditional — its head holds whenever its body is
   provable. Facts are ground truth; rules derive more from it.
2. **Query ↔ entailment.** `?- G.` asks Prolog to construct a proof of `G` from
   the clauses; success means $KB \vdash G$. Two caveats: it is derivability under
   SLD resolution specifically, and failure means only "no proof found", treated
   as false under the closed-world assumption.
3. **Why verify Python's plan with Prolog?** Because the two are **independent** —
   no shared code, data structures, or authorial assumptions. The Python planner
   could have a bug in `applicable` and Prolog would still refuse to prove
   `valid_move(a,c)`, because its answer derives from its own facts and engine.
   Agreement between independent systems is evidence; agreement between a program
   and its own explanation is not.
4. **What an independent verifier adds for LLM output.** It converts a
   *plausibility* judgement into a *derivability* one. This is the
   **generate-and-check** pattern, and its value comes from asymmetry: generating
   a plan is hard and error-prone, checking one is cheap and mechanical. That
   asymmetry is what lets an unreliable generator be used safely — provided the
   checker is genuinely independent, which is why it matters that Prolog runs in
   a separate process with its own knowledge base.

---

## 7. Task 5 — can the LLM verify its own plan?

Asked to justify its plan step by step, the LLM produced a fluent, correctly
formatted walkthrough with a state table after each action. For the **correct**
plan it was right — which is not the interesting case.

The interesting case is a plan that is *wrong but plausible*. Given the
handout's own sequence, an explanation writes itself:

> "Move(A,B) takes the robot to B. PickUp(Package,B) picks up the package at B.
> Move(B,C) carries it to C. Drop(Package,C) releases it, achieving
> At(Package,C)."

Every clause is locally plausible and the conclusion is exactly the goal. The
flaw is one unstated assumption — that the package moved with the robot — which
no amount of fluency exposes. The verifier finds it immediately: At(Package,B) is
simply not in the state.

### Which to trust: (a) the explanation, or (b) the executed transitions?

**(b), and not as a matter of degree — they are different kinds of object.**

An explanation is *generated text about* a computation, produced by the same
process with the same failure modes that produced the plan. If the plan is wrong
because a precondition was overlooked, the explanation is generated under that
same misapprehension and will confidently narrate the missing step. Fluency
carries no information about correctness.

Executed transitions are *the computation itself*. `verify_plan` re-derives each
state from the action definitions and tests each precondition as set containment.
It has no opinion and cannot be persuaded. Crucially it is **independent** of the
planner, sharing no code, so a bug in one cannot hide a bug in the other.

Note the asymmetry: the explanation must be *read carefully* to find the error;
the verifier reports it as a boolean. Only one of those scales.

**A generated explanation is not an independent verification.** It is a second
sample from the same distribution.

---

## 8. Reflection questions

**1. Why specify preconditions and effects before prompting?** Because they are
the *content* of the problem; everything else is scaffolding. BFS is generic —
the same twenty lines plan warehouse deliveries or solve a puzzle — so the action
schema is the only thing that makes this a warehouse planner. Without writing it
down first I would have delegated the actual modelling decision. It also makes
the output checkable: I could read the generated `Action` definitions against my
table. Without that table I could only ask whether the code looked reasonable —
and Test C2 showed that a planner missing its precondition check looks perfectly
reasonable and returns a *shorter* plan.

**2. An error from failing to check preconditions.** Exactly Test C2: the robot
drops a package it never held, in a room it never visited. The plan was one
action long, ran without error, and looked better than the correct one.

**3. Why "looks reasonable" ≠ valid.** Plausibility is a property of the
*sequence of action names*; validity is a property of the *state trajectory*. The
handout's own sequence has the right shape and is still invalid. Reading action
names cannot expose the gap because names do not encode state. This generalises:
any check operating on a plan's *description* rather than its *execution* can be
fooled by a plan whose description is well-formed.

**4. What the LLM contributed.** Dataclass boilerplate, the BFS skeleton, plan
printing, and the `subprocess` plumbing for calling SWI-Prolog non-interactively
— that last genuinely saved time. It did not contribute the domain model, the
decision to test a deliberately broken planner, or the subset goal test.

**5. What I verified independently.** All three corrections in Section 3 — and
they failed loudly, quietly, and invisibly respectively. Only the first would
have been caught by running the program; the second needed someone to question
the reported numbers, the third to read the code against the spec. Generated code
needs auditing at all three levels, because "it produces the right answer" is the
weakest of the three checks.

**6. Where logical reasoning is used.** Applicability ($S \models \text{Pre}(a)$,
entailment over ground propositions) and state update ($S' = (S \setminus
\text{Del}) \cup \text{Add}$, the semantics of what an action means). Prolog adds
a third and different kind: genuine *derivation* by SLD resolution — the step
from checking propositions to inferring new ones.

**7. Relation to the search module.** Planning **is** search over a different
state space:

| Search lab (grid) | This lab (planning) |
|---|---|
| state = `(row, col)` | state = set of ground propositions |
| actions = Up/Down/Left/Right | actions = Move / PickUp / Drop |
| successor = move if not a wall | successor = apply effects if preconditions hold |
| goal test = reached cell `G` | goal test = $G \subseteq S$ |
| BFS optimal (moves cost 1) | BFS optimal (actions cost 1) |

The real difference is where the successor function comes from: read off a map in
the grid, **computed by logical inference** here. That has consequences — the
grid's space is explicit and drawable; the planner's is defined implicitly and is
exponential in the number of propositions. Here it is 12 states and BFS copes.
Add ten packages and it explodes, which is why real planners use heuristics
derived from the logical structure (ignore-delete-lists relaxations, landmarks)
rather than blind BFS — the same move from blind to informed search that the
Search lab made with A\*.

---

## Takeaway

$$\textbf{Understand} \to \textbf{Specify} \to \textbf{Generate} \to \textbf{Execute} \to \textbf{Verify}$$

The specification in Task 0 is what made the generated code checkable; execution
in Task 3 is what exposed the broken planner; and independent verification — in
Python by replay, in Prolog by derivation — is what separates a plan that is
*valid* from one that merely *reads* well.

---

## Reference

S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. —
logical agents, STRIPS planning representations, and forward state-space search.
