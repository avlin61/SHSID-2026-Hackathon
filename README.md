# AI Data Center Power–Computing Co-Scheduling

**HackSHSID 2026 · Track 2** — how a greedy rule became a linear program, and then working code.

This README documents the reasoning chain we actually followed:

1. **Greedy rule** — what a human dispatcher would do (use free green first, then buy the cheapest hour).
2. **Math** — the same rule written as an optimization problem, plus a proof sketch that the greedy rule is exactly optimal here.
3. **Code** — how each symbol maps to `Code.py`, and the results of the current run.

---

## 0. TL;DR

| | |
|---|---|
| Fixed compute (must run in its hour) | **183.5 MWh/day**, from the input column |
| Flexible pool (schedulable) | **276.0 MWh/day**, from the parameters sheet |
| Total delivered | **459.5 MWh/day**, identical before and after |
| Electricity cost | **1,451,982 → 929,056 USD (−36.0%)** |
| Grid carbon | **1,436.3 → 1,137.6 tCO2 (−20.8%)** |
| Greedy rule vs LP optimum | **identical to the cent** (gap 0.000000 USD) |

The saving does **not** come from buying less electricity — total grid purchases are 243.1 MWh in both cases, and all available green is used in both. The saving comes from **when** the grid electricity is bought: the flexible pool is moved out of the expensive 17:00–21:00 window into the cheap early-morning hours, while the midday hours that still hold green surplus already sit at full capacity in both schedules.

---

## 1. What we are deciding

Each hour `t = 1…24` has:

| symbol | meaning | where it comes from |
|---|---|---|
| `F_t` | **fixed** compute that must run in hour `t` (online inference, latency-critical) | input column `Fixed compute (MW)` |
| `L_t` | non-compute facility load (cooling, network) | parameter, constant 3.86 MW |
| `G_t` | available green electricity | input column |
| `p_t` | electricity price | input column |
| `e_t` | grid carbon intensity | input column |
| `C_t` | total compute capacity | parameter, 23 MW |
| `x_t` | **flexible** compute we place in hour `t` | **decision variable** |
| `b_t` | grid electricity bought in hour `t` | derived |

The fixed part follows the working day — high during office hours, low at night — because that is when users are online. It is an **input**, not something the optimizer may move.

---

## 2. Step 0 — the greedy rule (our first intuition)

> *“Use the green electricity that is left after the fixed load first. When it runs out, look at the absolute cost of electricity + carbon in each hour and put the work in the cheapest one.”*

Written as a procedure:

```
1. serve the fixed load          : fixed + L against G, hour by hour
2. measure leftover green        : h_t = max(0, G_t − L_t − F_t)
3. pour flexible work into h_t   : free, marginal cost 0
4. still work left?              : sort the remaining hours by
                                   q_t = p_t + λ·e_t   (absolute cost, USD per MWh)
                                   and fill them cheapest-first,
                                   respecting the headroom C_t − F_t
```

Baseline for comparison: spread the same flexible work **as evenly as possible** over the 24 hours (capacity respected), no shifting.

This is deliberately simple enough to explain to a judge in thirty seconds. The next section shows it is not just a heuristic — it is the optimal solution of the scheduling problem.

---

## 3. Step 1 — the same rule as mathematics

### 3.1 Per-MWh prices

The input table gives **hourly totals** (USD/hour, tCO2/hour). To price one extra MWh we normalise by the renewable reference power:

```
price_intensity_t  = p_t / max(G_t, 1)      [USD per MWh]
carbon_intensity_t = e_t / max(G_t, 1)      [tCO2 per MWh]
```

(This is a modelling assumption, discussed in §7.)

### 3.2 Grid purchase

```
b_t = max(0, L_t + F_t + x_t − G_t)
```

Green is consumed first; anything above it is bought from the grid. This `max(·,·)` is what makes green “free” and creates the two-regime marginal cost in §3.4.

### 3.3 Optimization problem

```
minimize   Z = Σ_t ( price_t + λ · carbon_t ) · b_t

subject to Σ_t x_t = W                      all flexible work is delivered
           0 ≤ x_t ≤ C_t − F_t              per-hour capacity minus fixed load
           b_t = max(0, L_t + F_t + x_t − G_t)
           b_t ≥ 0
```

| symbol | value in the current run |
|---|---|
| `W` | 276 MWh/day |
| `C_t` | 23 MW (1 h slots, so 1 MW = 1 MWh) |
| `λ` | 50 USD/tCO2 |

With `λ = 0` the model is pure cost minimization; increasing `λ` puts more weight on carbon. The objective has the same form in both cases — carbon is simply priced. (In this dataset the ranking of hours happens to be unchanged, see §9.)

### 3.4 Why the greedy rule is optimal here

For one hour, the marginal cost of the next MWh of flexible work is:

```
0           while  x_t < h_t          (still inside leftover green)
q_t = price_t + λ·carbon_t   once  x_t > h_t   (must buy from the grid)
```

So the hourly cost is convex and piecewise-linear in `x_t`. The only coupling between hours is the single equality `Σ x_t = W`. Two exchange arguments finish the proof:

* If some hour still has unused leftover green while another hour is paying for grid power, moving 1 MWh into the free hour strictly lowers the cost.
* If two paid hours are used and the cheaper one still has headroom, moving 1 MWh from the expensive to the cheaper strictly lowers the cost.

Hence any allocation that violates the greedy order can be improved — the greedy allocation is a global optimum. (When the LP is solved directly, `scipy`/HiGHS reaches the same point: see §9.)

### 3.5 Linear-programming form used in the code

The `max()` is replaced by its standard epigraph form, which keeps the problem linear:

```
variables: x_t ≥ 0, b_t ≥ 0

minimize   Σ_t q_t · b_t
s.t.       Σ_t x_t = W
           x_t − b_t ≤ G_t − L_t − F_t        (b_t covers the shortfall)
           0 ≤ x_t ≤ C_t − F_t
```

---

## 4. From the first draft to the final model

Our first write-up used a simpler version where **all** compute was movable:

| | first draft (v0) | final model / code (v1) |
|---|---|---|
| grid purchase | `b_t = max(0, L_t + x_t − G_t)` | `b_t = max(0, L_t + F_t + x_t − G_t)` |
| workload | `Σ x_t = W`, `0 ≤ x_t ≤ C_t` | `Σ x_t = W_flex`, `0 ≤ x_t ≤ C_t − F_t` |
| horizon | `t = 1…3` generic slots | `t = 1…24` hourly slots |
| objective | `Σ p_t b_t`, then `q_t = p_t + λe_t` | same objective from the start |
| what moves | every MWh | only the flexible pool |

The reason for v1: inference that serves live users cannot be postponed. Treating it as movable produces an unrealistically good saving (75.8% instead of 36.0%) and an indefensible schedule — it would shut down the very hours when the data centre is busiest. Separating `F_t` (input) from `x_t` (decision) fixes the story and keeps the same objective.

---

## 5. Step 2 — from equations to code

| math | `Code.py` |
|---|---|
| `F_t`, `G_t`, `p_t`, `e_t` | `fixed`, `green`, `price`, `carbon` (section 1) |
| `W`, `C_t`, `L_t`, `λ` | read from the `Parameters` sheet (section 2) |
| `headroom = C_t − F_t` | `headroom = C − fixed` |
| `h_t = max(0, G_t − L_t − F_t)` | `surplus = np.maximum(0.0, green − L − fixed)` |
| `q_t = price_t + λ·carbon_t` | `q = price_intensity + lam * carbon_intensity` |
| objective `Σ q_t b_t` | `c[idx_b(t)] = q[t]` (section 5) |
| `Σ x_t = W` | `A_eq`, `b_eq` (section 6) |
| `x_t − b_t ≤ G_t − L_t − F_t` | `A_ub`, `b_ub` (section 7) |
| `0 ≤ x_t ≤ C_t − F_t` | `bounds` (section 8) |
| baseline | water-fill loop (section 10) |
| results | `SchedulingResults.xlsx`, three sheets (sections 13–14) |

The LP itself is ten lines of `scipy`:

```python
c = np.zeros(N)
for t in range(T):
    c[idx_b(t)] = q[t]                       # objective: only grid purchases are priced

A_eq[0, :T] = 1.0                            # Σ x_t = W
b_eq = np.array([W])

for t in range(T):
    A_ub[t, idx_x(t)] = 1.0
    A_ub[t, idx_b(t)] = -1.0
    b_ub[t] = green[t] - L[t] - fixed[t]     # x_t - b_t <= G_t - L_t - F_t

bounds = [(0.0, headroom[t]) for t in range(T)] + [(0.0, None) for _ in range(T)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
              bounds=bounds, method="highs")
```

The baseline deliberately does **not** simply use `W/24` (that would exceed capacity in the hours where fixed compute is high). It spreads the pool as evenly as possible while respecting `C_t − F_t`:

```python
left = W
active = np.ones(T, bool)
while left > 1e-9 and active.any():
    share = left / active.sum()
    capped = np.where(active & (headroom - x_base < share - 1e-9))[0]
    if len(capped) == 0:
        x_base[active] += share
        left = 0.0
    else:
        for t in capped:
            take = headroom[t] - x_base[t]
            x_base[t] += take
            left -= take
            active[t] = False
```

---

## 6. Results

**Totals**

| | baseline | optimized | change |
|---|---|---|---|
| fixed compute | 183.5 MWh | 183.5 MWh | — |
| flexible compute | 276.0 MWh | 276.0 MWh | — |
| electricity cost | 1,451,982.1 USD | 929,056.1 USD | **−36.0%** |
| grid carbon | 1,436.34 tCO2 | 1,137.62 tCO2 | **−20.8%** |
| grid purchased | 243.1 MWh | 243.1 MWh | 0 |
| green used | 309.0 MWh | 309.0 MWh | 0 |
| peak grid draw | 14.6 MW @ 19:00 | 19.9 MW @ 23:00 | +36% |
| cost per MWh delivered | 3,159.9 USD | 2,021.9 USD | −36.0% |

**Where the flexible pool goes**

| hour | `F_t` | baseline `x_t` | optimized `x_t` | why |
|---|---|---|---|---|
| 00:00–01:00 | 4.0 | 11.74 | **19.00** | cheapest absolute cost |
| 04:00–05:00 | 3.5 | 11.74 | **19.50** | cheapest absolute cost, headroom 19.5 |
| 10:00–11:00 | 12.5 | 10.50 | 10.50 | already at capacity in both |
| 19:00–20:00 | 7.0 | 11.74 | **0.00** | most expensive hour of the day |
| 21:00–22:00 | 5.0 | 11.74 | **0.00** | expensive |
| 23:00–00:00 | 4.0 | 11.74 | **19.00** | cheap again |

Only 46.7 MWh of the 276 MWh pool can ride on leftover green (16.9%); the rest must be bought, so its placement is decided by `q_t`. That is exactly the two-stage rule in §2.

---

## 7. Assumptions, caveats, and what we would add

* **Slots are 1 hour**, so 1 MW held for one hour = 1 MWh.
* **Price and carbon come as hourly totals**, and we convert them to per-MWh values by dividing by `G_t`. This matters: it makes cheap-but-dark nighttime hours look attractive. A fixed reference power (or an explicit USD/MWh column in the data) would remove the artefact and is a one-line change.
* **Perfect foresight**: we assume the day-ahead price, carbon intensity and green availability are known. In practice they are forecasts; the model can be re-run hourly with updated forecasts (rolling horizon).
* **No per-task deadlines**: flexibility is modelled as one daily pool `W`. Per-job deadlines would become additional constraints `Σ_{t ≤ d_j} x_j,t = W_j`.
* **No storage, no grid-connection limit.** The optimal schedule raises the peak grid draw from 14.6 MW to 19.9 MW. If the site has a firm interconnection limit, add `b_t ≤ B`; this is the first constraint we would add in a real deployment.
* **Carbon is priced, not constrained.** `λ = 50 USD/tCO2` is a policy choice; a hard cap would be `Σ e_t b_t ≤ E`.
* **Baseline is a fair "do nothing"**: the same total work, spread as evenly as capacity allows, with no shifting.

---

## 8. Files

| file | what it is |
|---|---|
| `ExampleData.xlsx` | input: sheet **Hourly** (24 rows × 5 columns) + sheet **Parameters** |
| `ExampleData_v1_backup.xlsx` | the original 6-column dataset, kept for reference |
| `Code.py` | model + solver + Excel writer |
| `SchedulingResults.xlsx` | output: **Summary**, **Allocation Plan**, **Before vs After** |
| `README.md` | this document |

**Input format**

`Hourly` (24 rows):

| Hour (Shanghai CST) | Fixed compute (MW) | Grid emissions (tCO2 per hour) | Electricity cost (USD per hour) | Renewable available (MW) |
|---|---|---|---|---|
| 00:00–01:00 | 4.0 | 41.296 | 27,056 | 8 |
| … | … | … | … | … |

`Parameters`:

| Parameter | Value |
|---|---|
| Daily flexible compute (MWh) | 276 |
| Compute capacity per hour (MW) | 23 |
| Base facility load (MW) | 3.86 |
| Carbon price lambda (USD/tCO2) | 50 |

**Output**

* **Summary** — fixed vs flexible split, before/after totals, headline.
* **Allocation Plan** — one row per hour: `F_t`, `x_t`, total compute, grid purchase, renewable used, `Green left after fixed`, price, carbon, action (`shift flexible in/out`).
* **Before vs After** — per-hour cost, carbon, flexible and grid purchase before/after, with money saved colour-coded green/red.

---

## 9. Reproduce

```bash
pip install pandas numpy scipy openpyxl
python3 Code.py
```

The console prints the per-hour allocation and the before/after table; the workbook is rewritten as `SchedulingResults.xlsx`.

**Why believe the result is (near-)optimal**

* The LP is solved to **proven optimality** by HiGHS (`linprog`, method `highs`); no heuristic is used in the final number.
* The greedy rule of §2 reproduces the LP allocation: cost gap **0.000000 USD**, largest per-hour difference **1e-13 MWh** (Appendix A reproduces the check).
* Capacity and total-work constraints are checked in the summary sheet (`capacity constraint satisfied = yes`, `flexible compute delivered = 276.0`).
* The `λ` sweep in `Code.py` section 12 re-solves the model for `λ ∈ {0, 25, 50, 100, 200}`. In this dataset the schedule does not change: price and carbon intensity rank the hours the same way, so the result is robust to the choice of `λ` (the sweep would diverge if cheap and clean hours disagreed).

---

## Appendix A — checking the greedy rule against the LP

Run this next to `ExampleData.xlsx`; it solves the model twice, once with the rule of §2 and once as an LP, and compares.

```python
import numpy as np, pandas as pd
from scipy.optimize import linprog

raw = pd.read_excel('ExampleData.xlsx', sheet_name='Hourly', header=1)
raw.columns = ['hour', 'fixed', 'emissions', 'cost', 'green']
par = (pd.read_excel('ExampleData.xlsx', sheet_name='Parameters', header=1)
         .dropna(subset=['Parameter']).set_index('Parameter')['Value'])

price, carbon, green, fixed = (raw[c].to_numpy(float) for c in ['cost', 'emissions', 'green', 'fixed'])
T = len(raw); G = np.maximum(green, 1.0)
p, e = price / G, carbon / G
W = float(par['Daily flexible compute (MWh)'])
C = np.full(T, float(par['Compute capacity per hour (MW)']))
L = np.full(T, float(par['Base facility load (MW)']))
lam = float(par['Carbon price lambda (USD/tCO2)'])
q = p + lam * e
head = C - fixed
h = np.maximum(0.0, green - L - fixed)

# greedy: free green first, then cheapest absolute cost
x = np.zeros(T); left = W
for t in np.argsort(-h, kind='stable'):
    take = min(left, head[t], h[t]); x[t] += take; left -= take
for t in np.argsort(q, kind='stable'):
    take = min(left, head[t] - x[t]); x[t] += take; left -= take
b = np.maximum(0.0, L + fixed + x - green)
greedy_cost = float(p @ b)

# same model as an LP
N = 2 * T
c = np.zeros(N); c[T:] = q
A_eq = np.zeros((1, N)); A_eq[0, :T] = 1
A_ub = np.zeros((T, N)); b_ub = np.zeros(T)
for t in range(T):
    A_ub[t, t] = 1; A_ub[t, T + t] = -1; b_ub[t] = green[t] - L[t] - fixed[t]
bounds = [(0, head[t]) for t in range(T)] + [(0, None)] * T
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[W], bounds=bounds, method='highs')
lp_cost = float(p @ res.x[T:])

print(f'greedy {greedy_cost:,.2f}   LP {lp_cost:,.2f}   gap {greedy_cost - lp_cost:.6f}')
```

Output on the shipped dataset:

```
greedy 929,056.07   LP 929,056.07   gap 0.000000
```

Same schedule to machine precision — the thirty-second rule and the optimization are the same answer.
