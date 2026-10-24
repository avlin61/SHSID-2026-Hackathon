# AI Data Center Power–Computing Co-Scheduling

**HackSHSID 2026 · Track 2**

A data center runs two kinds of AI work: **fixed** compute that must run in its own hour (online inference), and a **flexible** daily pool (batch training) that can be placed anywhere. This program decides where to put the flexible pool so electricity cost and grid carbon are as low as possible, without ever moving the fixed part.

---

## 1. How we got there — three steps

### Step 1 — the greedy rule (our first idea)

For every hour we know the electricity price, the grid carbon intensity, the available green power and the fixed compute. A human dispatcher would do:

1. Run the fixed load first. Whatever green power is left over is **free** for flexible work.
2. Pour flexible work into that leftover green first — it costs nothing.
3. When the free green is used up, rank the remaining hours by **absolute cost** `price + λ × carbon` and fill the cheapest ones, up to each hour's remaining capacity.

```
for each hour t:  h_t = max(0, G_t − L_t − F_t)        # leftover green
fill x_t with h_t first                                # free
then sort the rest by q_t = p_t + λ·e_t and fill       # cheapest absolute cost
never exceed C_t − F_t                                 # headroom
```

### Step 2 — the same rule written as formulas

```
grid purchase   b_t = max(0, L_t + F_t + x_t − G_t)
absolute cost   q_t = p_t + λ · e_t

minimize        Σ_t q_t · b_t
subject to      Σ_t x_t = W                  (all flexible work is delivered)
                0 ≤ x_t ≤ C_t − F_t          (capacity minus fixed load)
```

The two rules are the same thing: the marginal cost of the next MWh is **0** while the hour still has leftover green, and **q_t** after that. Moving work out of a paid hour into a free or cheaper one always lowers the bill, so the greedy order is an optimum — and when we solve the formulas as a linear program we get exactly the same allocation (cost gap 0.000000 USD).

> Our first version treated all 276 MWh as movable (`b_t = max(0, L_t + x_t − G_t)`, saving 75.8%). That implied shutting down inference in the busiest hours, so we split the compute into fixed `F_t` (input) and flexible `W` (decision). Same objective, defensible schedule, saving 36.0%.

### Step 3 — into code

| in the formulas | in `Code.py` |
|---|---|
| `F_t, G_t, p_t, e_t` | columns of the `Hourly` sheet |
| `W, C_t, L_t, λ` | `Parameters` sheet |
| `b_t = max(0, L+F+x−G)` | `A_ub`, `b_ub`, bounds |
| `Σ x_t = W` | `A_eq`, `b_eq` |
| `min Σ q_t b_t` | objective vector, `q = price_intensity + lam·carbon_intensity` |
| solve | `scipy.optimize.linprog(..., method="highs")` |
| baseline | same flexible pool spread as evenly as capacity allows |
| output | three sheets of `SchedulingResults.xlsx` |

---

## 2. Input

`ExampleData.xlsx`, two sheets. All prices and emissions are **hourly totals**, which the code converts to per-MWh values by dividing by the renewable reference power.

**Hourly** (24 rows)

| Hour (Shanghai CST) | Fixed compute (MW) | Grid emissions (tCO2 per hour) | Electricity cost (USD per hour) | Renewable available (MW) |
|---|---|---|---|---|
| 00:00–01:00 | 4.0 | 41.296 | 27,056 | 8 |
| … | … | … | … | … |
| 10:00–11:00 | 12.5 | 73.830 | 83,059 | 20 |

The fixed compute follows the working day — high during office hours, low at night — and is editable directly in the sheet.

**Parameters**

| Parameter | Value |
|---|---|
| Daily flexible compute (MWh) | 276 |
| Compute capacity per hour (MW) | 23 |
| Base facility load (MW) | 3.86 |
| Carbon price lambda (USD/tCO2) | 50 |

---

## 3. Output

`SchedulingResults.xlsx` is written by `Code.py` (three sheets), and the same table is printed to the console.

**Summary** — fixed vs flexible split, then the before/after totals (cost, carbon, grid purchase, green used, peak values, cost per MWh) and a one-line headline.

**Allocation Plan** — one row per hour, the actual schedule:

| column | meaning |
|---|---|
| Fixed compute `F_t` | must-run compute in that hour (from the input) |
| Flexible allocated `x_t` | what the optimizer places there |
| Total compute | `F_t + x_t` |
| Share / cumulative | how the flexible pool is spread over the day |
| Grid purchase `b_t` | what has to be bought that hour |
| Grid-free? / Green left after fixed | shows where the leftover green goes |
| Price, carbon, G, L, C | the inputs that drove the decision |
| Action | `shift flexible in` / `shift flexible out` vs the baseline |

**Before vs After** — per-hour comparison: cost and carbon before/after, money saved, flexible and grid purchase before/after, colour-coded (green = saved, red = deliberately spent more).

---

## 4. Results

| | no scheduling | optimized |
|---|---|---|
| fixed compute | 183.5 MWh | 183.5 MWh |
| flexible compute | 276.0 MWh | 276.0 MWh |
| electricity cost | 1,451,982 USD | **929,056 USD (−36.0%)** |
| grid carbon | 1,436.34 tCO2 | **1,137.62 tCO2 (−20.8%)** |
| grid purchased | 243.1 MWh | 243.1 MWh |

The saving is not "buy less electricity" — total purchases are identical. It is **when** the electricity is bought: flexible work leaves the expensive 17:00–21:00 window and moves into cheap early-morning hours (00:00–07:00), while midday hours that still hold leftover green stay at full capacity.

---

## 5. Files and how to run

| file | what it is |
|---|---|
| `Code.py` | reads the input, solves the model, writes the output |
| `ExampleData.xlsx` | input (Hourly + Parameters) |
| `ExampleData_v1_backup.xlsx` | the original 6-column dataset, before the fixed/flexible split |
| `SchedulingResults.xlsx` | generated output (not committed — run the code to create it) |

```bash
pip install pandas numpy scipy openpyxl
python3 Code.py
```

The console prints the per-hour allocation and the before/after table, and the workbook is written next to `Code.py`.
