import pandas as pd
import numpy as np
from scipy.optimize import linprog


# 1. read excel  (sheet "Hourly": one row per hour; sheet "Parameters": the daily knobs)
file='ExampleData.xlsx'#here you might need to use an absolute path, else may occur filenotfounderror
raw=pd.read_excel(file,sheet_name='Hourly',header=1)

assert raw.shape[1]==5,f'expect 5 columns, actual {raw.shape[1]} columns'

raw.columns=['hour','fixed_MW','emissions','cost_usd','renewable_MW']
raw=raw.dropna(subset=['hour']).reset_index(drop=True)
raw=raw.dropna(subset=['fixed_MW','emissions','cost_usd','renewable_MW'])

T=len(raw) #24

assert T==24, f'Expected 24 rows, got {T}' #自检行数

# daily knobs live on the Parameters sheet
par=pd.read_excel(file,sheet_name='Parameters',header=1)
par=par.dropna(subset=['Parameter']).set_index('Parameter')['Value']

#put into numpy arrays
price=raw['cost_usd'].astype(float).values
carbon=raw['emissions'].astype(float).values
green=raw['renewable_MW'].astype(float).values
fixed=raw['fixed_MW'].astype(float).values


P_ref=np.maximum(green,1.0)#参考功率，renewable MW
price_intensity=price/P_ref#USD/MWh
carbon_intensity=carbon/P_ref#tCO2/MWh

assert not np.isnan(price_intensity).any()#自检无nan
assert not np.isnan(carbon_intensity).any()

print(f'T={T}')
print(f'fixed compute range:{fixed.min():.1f}-{fixed.max():.1f}MW')
print(f'price range:{price.min():.0f}-{price.max():.0f}USD/hour')
print(f'carbon range:{carbon.min():.1f}-{carbon.max():.1f}tCO2/hour')
print(f'green range:{green.min():.1f}-{green.max():.1f}MW')

#2. model parameters (read from the Parameters sheet)

W=float(par['Daily flexible compute (MWh)'])#MWh of flexible compute to place somewhere in the day
C=np.full(T,float(par['Compute capacity per hour (MW)']))#MW total compute cap per hour
L=np.full(T,float(par['Base facility load (MW)']))#MW non-compute load (cooling, network, ...)
lam=float(par['Carbon price lambda (USD/tCO2)'])

fixed_energy=float(fixed.sum())
total_energy=fixed_energy+W
headroom=C-fixed#MW left for flexible compute
surplus=np.maximum(0.0,green-L-fixed)#MW of green left after the fixed compute is served

assert np.all(fixed<=C+1e-9),'fixed compute exceeds capacity in some hour'
assert float(headroom.sum())>=W-1e-9,'not enough compute headroom for the flexible pool'

print(f'flexible W={W:.1f}MWh  fixed={fixed_energy:.1f}MWh  total={total_energy:.1f}MWh')
print(f'green left for flexible after fixed load: {surplus.sum():.1f}MWh')

#3. Combined cost
q=price_intensity+lam*carbon_intensity#USD?MWh

#4. Decision variables
N=2*T
idx_x=lambda t:t
idx_b=lambda t:T+t

#5. min sum
c=np.zeros(N)
for t in range(T):
    c[idx_b(t)]=q[t]

#6. equality constraint: sum_t x_t=W
A_eq=np.zeros((1,N))
for t in range(T):
    A_eq[0,idx_x(t)]=1.0
b_eq=np.array([W])

#7. inequality constraint
A_ub,b_ub=[],[]
for t in range(T):
    row=np.zeros(N)
    row[idx_x(t)]=1.0
    row[idx_b(t)]=-1.0
    A_ub.append(row)
    b_ub.append(green[t]-L[t]-fixed[t])

A_ub=np.array(A_ub)
b_ub=np.array(b_ub)

#8. variable bounds
bounds=[]
for t in range(T):
    bounds.append((0.0,headroom[t]))
for t in range(T):
    bounds.append((0.0,None))

#9. Solve
res=linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method='highs')
if not res.success:
    print('solver failed:',res.message)
    raise SystemExit

x_sol=res.x[:T]
b_sol=res.x[T:]

#10. Baseline: the same flexible work spread as evenly as possible (capacity still respected)
x_base=np.zeros(T)
left=W
active=np.ones(T,bool)
while left>1e-9 and active.any():
    share=left/active.sum()
    capped=np.where(active&(headroom-x_base<share-1e-9))[0]
    if len(capped)==0:
        x_base[active]+=share
        left=0.0
    else:
        for t in capped:
            take=headroom[t]-x_base[t]
            x_base[t]+=take
            left-=take
            active[t]=False
b_base=np.maximum(0.0,L+fixed+x_base-green)

#11. Evaluate metrics
def metrics(x,b):
    total_cost=np.sum(price_intensity*b)
    total_carbon=np.sum(carbon_intensity*b)
    work_done=np.sum(x)
    return total_cost,total_carbon,work_done

cost_base,carbon_base,work_base=metrics(x_base,b_base)
cost_opt, carbon_opt, work_opt=metrics(x_sol,b_sol)

#12. lambda sensitivity sweep AI GEN

lam_values = [0.0, 25.0, 50.0, 100.0, 200.0]
sens_rows = []
for lam_test in lam_values:
    q_test = price_intensity + lam_test * carbon_intensity
    c_test = np.zeros(N)
    for t in range(T):
        c_test[idx_b(t)] = q_test[t]
    r = linprog(c_test, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs")
    if r.success:
        xt, bt = r.x[:T], r.x[T:]
        ct, et, _ = metrics(xt, bt)
        sens_rows.append({
            "lambda (USD/tCO2)": lam_test,
            "Total cost (USD)": round(float(ct), 1),
            "Total carbon (tCO2)": round(float(et), 2),
            "q*b objective": round(float(np.sum(q_test * bt)), 1),
        })
    else:
        sens_rows.append({
            "lambda (USD/tCO2)": lam_test,
            "Total cost (USD)": "infeasible",
            "Total carbon (tCO2)": "infeasible",
            "q*b objective": "infeasible",
        })

print()
print('lambda sweep (cost vs carbon trade-off):')
for row in sens_rows:
    print('  ', row)

#13. Output tables (presentation only)

# per-slot money and emissions, before vs after (1-hour slots, so MW == MWh)
cost_slot_base=price_intensity*b_base
cost_slot_opt=price_intensity*b_sol
carbon_slot_base=carbon_intensity*b_base
carbon_slot_opt=carbon_intensity*b_sol

renew_used_base=np.minimum(green,L+fixed+x_base)
renew_used_opt=np.minimum(green,L+fixed+x_sol)
free_green_base=float(np.minimum(x_base,surplus).sum())#flexible MWh that ride on leftover green
free_green_opt=float(np.minimum(x_sol,surplus).sum())

compute_base=fixed+x_base
compute_opt=fixed+x_sol

delta_x=x_sol-x_base
delta_cost=cost_slot_opt-cost_slot_base
delta_carbon=carbon_slot_opt-carbon_slot_base

# kill floating point noise so the tables never show "-0"
tiny=1e-6
delta_x=np.where(np.abs(delta_x)<tiny,0.0,delta_x)
delta_cost=np.where(np.abs(delta_cost)<tiny,0.0,delta_cost)
delta_carbon=np.where(np.abs(delta_carbon)<tiny,0.0,delta_carbon)
saved_cost=np.where(np.abs(delta_cost)<tiny,0.0,-delta_cost)
saved_carbon=np.where(np.abs(delta_carbon)<tiny,0.0,-delta_carbon)

hour_arr=raw['hour'].values
action=np.where(delta_x>tiny,'shift flexible in',np.where(delta_x<-tiny,'shift flexible out','unchanged'))
grid_free=np.where((L+fixed+x_sol)<=green+1e-6,'yes','no')

share=x_sol/W*100
cum=np.cumsum(x_sol)

def pct(base,opt):
    return round((opt-base)/base*100,1) if abs(base)>1e-9 else ''

base_grid=float(b_base.sum());         opt_grid=float(b_sol.sum())
base_ren=float(renew_used_base.sum()); opt_ren=float(renew_used_opt.sum())
base_del=float((L+fixed+x_base).sum());opt_del=float((L+fixed+x_sol).sum())
base_zero=int(np.sum(b_base<1e-6));    opt_zero=int(np.sum(b_sol<1e-6))
b_peak_b=int(np.argmax(b_base));       b_peak_o=int(np.argmax(b_sol))
t_peak_b=int(np.argmax(compute_base)); t_peak_o=int(np.argmax(compute_opt))

#sheet1: summary - fixed vs flexible split, then before/after
summary=pd.DataFrame({'metric':['lambda (USD/tCO2)',
                                'fixed compute F_t, from input (MWh/day)',
                                'flexible compute W, to be placed (MWh/day)',
                                'total compute (MWh/day)',
                                'total electricity cost (USD)',
                                'total carbon emissions (tCO2)',
                                'grid electricity purchased (MWh)',
                                'renewable electricity used (MWh)',
                                'grid share of delivered energy (%)',
                                'green headroom for flexible (MWh/day)',
                                'flexible placed on free green (MWh)',
                                'peak grid purchase (MW @ hour)',
                                'peak total compute (MW @ hour)',
                                'cost per MWh delivered (USD/MWh)',
                                'carbon per MWh delivered (tCO2/MWh)',
                                'flexible compute delivered (MWh)',
                                'capacity constraint satisfied'],
                      'no scheduling (baseline)':[lam,round(fixed_energy,1),round(W,1),round(total_energy,1),
                                                  round(cost_base,1),round(carbon_base,2),
                                                  round(base_grid,1),round(base_ren,1),round(base_grid/base_del*100,1),round(float(surplus.sum()),1),round(free_green_base,1),
                                                  f'{b_base[b_peak_b]:.1f} @ {hour_arr[b_peak_b]}',
                                                  f'{compute_base[t_peak_b]:.1f} @ {hour_arr[t_peak_b]}',
                                                  round(cost_base/total_energy,1),round(carbon_base/total_energy,3),round(work_base,1),
                                                  'yes' if np.all(compute_base<=C+1e-6) else 'no'],
                      'optimized schedule':[lam,round(fixed_energy,1),round(W,1),round(total_energy,1),
                                            round(cost_opt,1),round(carbon_opt,2),
                                            round(opt_grid,1),round(opt_ren,1),round(opt_grid/opt_del*100,1),round(float(surplus.sum()),1),round(free_green_opt,1),
                                            f'{b_sol[b_peak_o]:.1f} @ {hour_arr[b_peak_o]}',
                                            f'{compute_opt[t_peak_o]:.1f} @ {hour_arr[t_peak_o]}',
                                            round(cost_opt/total_energy,1),round(carbon_opt/total_energy,3),round(work_opt,1),
                                            'yes' if np.all(compute_opt<=C+1e-6) else 'no'],
                      'Change (%)':['',0.0,0.0,0.0,pct(cost_base,cost_opt),pct(carbon_base,carbon_opt),
                                    pct(base_grid,opt_grid),pct(base_ren,opt_ren),pct(base_grid/base_del,opt_grid/opt_del),
                                    0.0,0.0,'','',pct(cost_base/total_energy,cost_opt/total_energy),
                                    pct(carbon_base/total_energy,carbon_opt/total_energy),0.0,'']})

headline_text=(f"Headline: moving the {W:,.1f} MWh flexible pool saves USD {cost_base-cost_opt:,.0f} "
               f"({pct(cost_base,cost_opt):+.1f}%) and {carbon_base-carbon_opt:,.1f} tCO2 "
               f"({pct(carbon_base,carbon_opt):+.1f}%); the {fixed_energy:,.1f} MWh fixed compute never moves.")

#sheet2: concrete allocation plan, one row per hour
alloc=pd.DataFrame({'Hour index':np.arange(T),
                    'Hour (Shanghai CST)':hour_arr,
                    'Fixed compute F_t (MW)':np.round(fixed,2),
                    'Flexible allocated x_t (MW)':np.round(x_sol,2),
                    'Total compute (MW)':np.round(compute_opt,2),
                    'Share of flexible pool (%)':np.round(share,1),
                    'Cumulative flexible (MWh)':np.round(cum,1),
                    'Grid purchase b_t (MW)':np.round(b_sol,2),
                    'Renewable used (MW)':np.round(renew_used_opt,2),
                    'Grid-free?':grid_free,
                    'Change vs flat flexible (MW)':np.round(delta_x,2),
                    'Action':action,
                    'Price p_t (USD/MWh)':np.round(price_intensity,2),
                    'Carbon e_t (tCO2/MWh)':np.round(carbon_intensity,3),
                    'Renewable available G_t (MW)':np.round(green,1),
                    'Green left after fixed (MW)':np.round(surplus,2),
                    'Base load L_t (MW)':np.round(L,1),
                    'Capacity C_t (MW)':np.round(C,1)})
alloc.loc[len(alloc)]=['TOTAL','24 h',
                       round(fixed_energy,2),round(float(x_sol.sum()),2),round(total_energy,2),100.0,round(float(cum[-1]),1),
                       round(opt_grid,2),round(opt_ren,2),f'{opt_zero} hours',
                       round(float(delta_x.sum()),2),'','','','','','','']

#sheet3: before/after consumption comparison, one row per hour
compare=pd.DataFrame({'Hour index':np.arange(T),
                      'Hour (Shanghai CST)':hour_arr,
                      'Baseline cost (USD)':np.round(cost_slot_base,1),
                      'Optimized cost (USD)':np.round(cost_slot_opt,1),
                      'Cost saved (USD)':np.round(saved_cost,1),
                      'Baseline carbon (tCO2)':np.round(carbon_slot_base,2),
                      'Optimized carbon (tCO2)':np.round(carbon_slot_opt,2),
                      'Carbon saved (tCO2)':np.round(saved_carbon,2),
                      'Baseline flexible (MW)':np.round(x_base,2),
                      'Optimized flexible (MW)':np.round(x_sol,2),
                      'Flexible shifted (MW)':np.round(delta_x,2),
                      'Baseline total compute (MW)':np.round(compute_base,2),
                      'Optimized total compute (MW)':np.round(compute_opt,2),
                      'Baseline grid purchase (MW)':np.round(b_base,2),
                      'Optimized grid purchase (MW)':np.round(b_sol,2),
                      'Grid purchase change (MW)':np.round(b_sol-b_base,2),
                      'Action':action})
compare.loc[len(compare)]=['TOTAL','24 h',
                           round(float(cost_slot_base.sum()),1),round(float(cost_slot_opt.sum()),1),round(float(saved_cost.sum()),1),
                           round(float(carbon_slot_base.sum()),2),round(float(carbon_slot_opt.sum()),2),round(float(saved_carbon.sum()),2),
                           round(float(x_base.sum()),2),round(float(x_sol.sum()),2),round(float(delta_x.sum()),2),
                           round(float(compute_base.sum()),2),round(float(compute_opt.sum()),2),
                           round(float(b_base.sum()),2),round(float(b_sol.sum()),2),round(float((b_sol-b_base).sum()),2),'']

#14. output excel

outputfile='SchedulingResults.xlsx'
with pd.ExcelWriter(outputfile,engine='openpyxl')as writer:
    summary.to_excel(writer,sheet_name='Summary',index=False,startrow=1)
    alloc.to_excel(writer,sheet_name='Allocation Plan',index=False,startrow=1)
    compare.to_excel(writer,sheet_name='Before vs After',index=False,startrow=1)

    from openpyxl.styles import Font,Alignment,PatternFill,Border,Side
    from openpyxl.utils import get_column_letter

    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill("solid", fgColor="2F5597")
    title_font = Font(bold=True, size=14, color="2F5597")
    total_font = Font(bold=True, size=11, color="1F3864")
    note_font = Font(bold=True, size=12, color="1F3864")
    small_font = Font(italic=True, size=10, color="595959")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    note_align = Alignment(horizontal="left", vertical="center", wrap_text=True)
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    good_fill = PatternFill("solid", fgColor="E2EFDA")
    bad_fill = PatternFill("solid", fgColor="FCE4D6")

    sheet_list=["Summary","Allocation Plan","Before vs After"]
    for sheet_name in sheet_list:
        ws = writer.sheets[sheet_name]

        ws.cell(row=1, column=1, value=f"AI Data Center Scheduling — {sheet_name}").font = title_font
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=2, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center
            cell.border = border

        # Body cells
        for row_idx in range(3, ws.max_row + 1):
            for col_idx in range(1, ws.max_column + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.alignment = center
                cell.border = border

        # Auto column widths
        for col_idx in range(1, ws.max_column + 1):
            letter = get_column_letter(col_idx)
            max_len = 0
            for row_idx in range(2, ws.max_row + 1):
                val = ws.cell(row=row_idx, column=col_idx).value
                if val is not None:
                    max_len = max(max_len, len(str(val)))
            ws.column_dimensions[letter].width = min(max_len + 4, 32)

        # Freeze panes below header
        ws.freeze_panes = "A3"

    # number formats so the sheets read like a plan, not a raw dump
    def set_fmt(ws,cols,code):
        for r in range(3, ws.max_row + 1):
            for c in cols:
                cell = ws.cell(row=r, column=c)
                if isinstance(cell.value,(int,float)):
                    cell.number_format = code

    ws_sum = writer.sheets['Summary']
    set_fmt(ws_sum,[2,3],'#,##0.##'); set_fmt(ws_sum,[4],'+0.0;-0.0;0.0')

    ws_alloc = writer.sheets['Allocation Plan']
    set_fmt(ws_alloc,[3,4,5,8,9,11],'0.00'); set_fmt(ws_alloc,[6,7],'0.0')
    set_fmt(ws_alloc,[13],'#,##0.0'); set_fmt(ws_alloc,[14],'0.000'); set_fmt(ws_alloc,[15,16,17,18],'0.00')

    ws_cmp = writer.sheets['Before vs After']
    set_fmt(ws_cmp,[3,4,5],'#,##0.0'); set_fmt(ws_cmp,[6,7,8],'#,##0.00'); set_fmt(ws_cmp,[9,10,11,12,13,14,15,16],'0.00')

    # green = saved, red = paid extra (before/after comparison at a glance)
    for r in range(3, ws_cmp.max_row + 1):
        for c in (5,8):
            cell = ws_cmp.cell(row=r, column=c)
            if isinstance(cell.value,(int,float)):
                cell.fill = good_fill if cell.value >= 0 else bad_fill

    # bold the TOTAL row on every table
    for ws in (ws_alloc, ws_cmp):
        for c in range(1, ws.max_column + 1):
            ws.cell(row=ws.max_row, column=c).font = total_font

    # explanatory notes under the comparison table
    note_cmp = ws_cmp.max_row + 2
    ws_cmp.cell(row=note_cmp, column=1,
                value=('Green = money / carbon saved in that hour vs the flat baseline; red = the optimizer deliberately spends more there '
                       '(cheap hours, including buying grid power) so the expensive hours can be cut. '
                       'Fixed compute F_t is identical before and after - only the flexible pool moves.')).font = small_font
    ws_cmp.cell(row=note_cmp, column=1).alignment = note_align
    ws_cmp.row_dimensions[note_cmp].height = 34
    ws_cmp.merge_cells(start_row=note_cmp, start_column=1, end_row=note_cmp, end_column=8)

    # headline + method note under the summary table
    ws_sum = writer.sheets['Summary']
    note_row = ws_sum.max_row + 2
    ws_sum.cell(row=note_row, column=1, value=headline_text).font = note_font
    ws_sum.cell(row=note_row, column=1).alignment = note_align
    ws_sum.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=4)
    ws_sum.row_dimensions[note_row].height = 24
    ws_sum.cell(row=note_row + 1, column=1,
                value=('Baseline = the flexible pool spread as evenly as possible over the 24 h (capacity respected) on top of the fixed compute, nothing shifted. '
                       'Optimized = fill the green left after the fixed load first (that part costs nothing), then use the hours with the lowest '
                       'absolute cost price + lambda*carbon. Slots are 1 h, so 1 MW held for one hour = 1 MWh.')).font = small_font
    ws_sum.cell(row=note_row + 1, column=1).alignment = note_align
    ws_sum.row_dimensions[note_row + 1].height = 42
    ws_sum.merge_cells(start_row=note_row + 1, start_column=1, end_row=note_row + 1, end_column=4)

    # console version of the allocation list + before/after diff
    print()
    print('=' * 100)
    print(f'BEFORE -> AFTER   cost {cost_base:,.0f} -> {cost_opt:,.0f} USD ({pct(cost_base,cost_opt):+.1f}%)  |  '
          f'carbon {carbon_base:,.1f} -> {carbon_opt:,.1f} tCO2 ({pct(carbon_base,carbon_opt):+.1f}%)  |  '
          f'fixed {fixed_energy:,.1f} MWh + flexible {W:,.1f} MWh')
    print('=' * 100)
    print(f"{'hour':<14}{'F fixed':>8}{'x before':>9}{'x after':>9}{'b before':>9}{'b after':>9}"
          f"{'cost before':>13}{'cost after':>12}{'USD saved':>11}{'tCO2 saved':>11}")
    for i in range(T):
        print(f"{str(hour_arr[i]):<14}"
              f"{fixed[i]:8.2f}{x_base[i]:9.2f}{x_sol[i]:9.2f}{b_base[i]:9.2f}{b_sol[i]:9.2f}"
              f"{cost_slot_base[i]:13,.0f}{cost_slot_opt[i]:12,.0f}{saved_cost[i]:11,.0f}{saved_carbon[i]:11.1f}")
    print('-' * 100)
    print(f"{'TOTAL':<14}{fixed_energy:8.2f}{x_base.sum():9.2f}{x_sol.sum():9.2f}{b_base.sum():9.2f}{b_sol.sum():9.2f}"
          f"{cost_slot_base.sum():13,.0f}{cost_slot_opt.sum():12,.0f}{saved_cost.sum():11,.0f}{saved_carbon.sum():11,.1f}")
    print()
    print('Sheets: Summary | Allocation Plan | Before vs After')
    print('Task Finished. Result written to SchedulingResults.xlsx')
