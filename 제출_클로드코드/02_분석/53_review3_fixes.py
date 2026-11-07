# -*- coding: utf-8 -*-
"""53_review3_fixes.py — 3차 심사 지적 재계산·근거 보존.

(1) 표본특성(Table S6·Figure 1)을 available-case 분모로 재산출 — 결측을 '해당없음'으로 세던 오류
(2) 부트스트랩 검증(설계기반 SE 대조)을 스크립트로 보존 + 결과 파일화 (재현 불가 지적 해소)
(3) Figure 5(SHAP)를 학습셋 기반 shap_importance_train.csv로 재생성

출력: results/characteristics_availablecase.csv, results/bootstrap_validation.csv, results/figures/shap_importance.png
"""
import sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = r"C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model"; RES = f"{D}/results"; FIG = f"{RES}/figures"; SEED = 42
kchs = pd.read_parquet(f"{D}/data/ext_kchs.parquet"); pre = pd.read_parquet(f"{D}/data/ext_pre.parquet"); post = pd.read_parquet(f"{D}/data/ext_post.parquet")

# ══════ (1) available-case 표본특성 ══════
print("="*86); print("[1] 표본특성 — available-case 분모(결측 제외) 재산출"); print("="*86, flush=True)
BINARY = [("Depression PHQ-9 >=10","dep"),("Female",None),("Current smoker",None),("Partnered","partnered"),
          ("Employed","employed"),("Food insecure","food_insecure"),("Hypertension","htn"),("Diabetes","dm")]
CONT = [("Age (years)","age"),("BMI","bmi"),("Sleep hours","sleep_hrs"),("Self-rated health","srh"),("Oral health","oral_health")]
rows=[]
def wmean(v, w):
    m = v.notna().values
    if m.sum()==0: return np.nan, 1.0
    return float(np.average(v[m].astype(float).values, weights=w[m])), float(1-m.mean())
for label, col in BINARY:
    r={"characteristic":label}
    for nm,d in [("KCHS",kchs),("NHANES pre",pre),("NHANES post",post)]:
        w=d["W"].to_numpy(float)
        if label=="Female":  v=(d["sex"]=="F").astype(float); v[d["sex"].isna()]=np.nan
        elif label=="Current smoker": v=pd.to_numeric(d["smoking"].map({"current":1,"former":0,"never":0}),errors="coerce")
        else: v=pd.to_numeric(d[col],errors="coerce")
        m,miss=wmean(pd.Series(v),w)
        r[nm]=round(m*100,2); r[nm+" missing%"]=round(miss*100,2)
    rows.append(r)
for label,col in CONT:
    r={"characteristic":label}
    for nm,d in [("KCHS",kchs),("NHANES pre",pre),("NHANES post",post)]:
        m,miss=wmean(pd.to_numeric(d[col],errors="coerce"),d["W"].to_numpy(float))
        r[nm]=round(m,2); r[nm+" missing%"]=round(miss*100,2)
    rows.append(r)
ch=pd.DataFrame(rows); ch.to_csv(f"{RES}/characteristics_availablecase.csv",index=False,encoding="utf-8-sig")
print(ch.to_string(index=False), flush=True)

# ══════ (2) 부트스트랩 검증 보존 ══════
print("\n"+"="*86); print("[2] 재척도 부트스트랩 검증 — 설계기반(선형화 ultimate-cluster) SE와 대조"); print("="*86, flush=True)
src=open(f"{D}/47_review2_recompute.py",encoding="utf8").read()
ns={"np":np,"pd":pd}; exec(src[src.index("class RaoWuBoot"):src.index("def boot_ci")],ns); RaoWuBoot=ns["RaoWuBoot"]
def design_se(d):
    y=d["dep"].astype(int).values; w=d["W"].to_numpy(float)
    p=np.average(y,weights=w); z=w*(y-p)
    g=pd.DataFrame({"st":d["strata"].astype(str).values,"psu":d["psu"].astype(str).values,"z":z}).groupby(["st","psu"])["z"].sum().reset_index()
    var=0.0
    for _,grp in g.groupby("st"):
        n=len(grp)
        if n<2: continue
        var+= n/(n-1)*((grp["z"]-grp["z"].mean())**2).sum()
    return float(np.sqrt(var)/w.sum())
vrows=[]
for nm,d in [("NHANES pre",pre),("NHANES post",post)]:
    y=d["dep"].astype(int).values
    dse=design_se(d)
    rw=RaoWuBoot(d["strata"].values,d["psu"].values,d["W"].to_numpy(float))
    rng=np.random.RandomState(1); est=[]
    for _ in range(4000):
        w=rw.weights(rng); s=w.sum()
        if s>0: est.append(float(np.dot(w,y)/s))
    est=np.array(est); bse=float(est.std(ddof=1))
    print(f"   {nm:<12} 설계기반 SE {dse:.5f} · 부트스트랩 SE {bse:.5f} · 비 {bse/dse:.3f}", flush=True)
    vrows.append(dict(cohort=nm,estimand="weighted depression prevalence",design_SE=round(dse,5),
                      bootstrap_SE=round(bse,5),ratio=round(bse/dse,3),replicates=len(est)))
pd.DataFrame(vrows).to_csv(f"{RES}/bootstrap_validation.csv",index=False,encoding="utf-8-sig")

# ══════ (3) Figure 5 — 학습셋 기반 SHAP ══════
print("\n"+"="*86); print("[3] Figure 5 — 학습셋 기반 SHAP 중요도로 재생성"); print("="*86, flush=True)
imp=pd.read_csv(f"{RES}/shap_importance_train.csv",index_col=0).iloc[:,0].sort_values()
NAMES={"srh":"Self-rated health","age":"Age","sex":"Sex","sleep_hrs":"Sleep duration","oral_health":"Self-rated oral health",
       "employed":"Employment","partnered":"Partnership","smoking":"Smoking","bmi":"BMI","food_insecure":"Food insecurity",
       "income_q":"Household income","current_drinker":"Current drinking","edu":"Education","living_alone":"Living alone",
       "active":"Physical activity","htn":"Hypertension","dm":"Diabetes"}
ADDED={"employed","oral_health","food_insecure"}
fig,ax=plt.subplots(figsize=(7.4,6.2))
cols=["#c0392b" if i in ADDED else "#2E5B9A" for i in imp.index]
ax.barh([NAMES.get(i,i) for i in imp.index], imp.values, color=cols)
for yy,(i,v) in enumerate(imp.items()): ax.text(v+0.006, yy, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("Mean |SHAP| (KCHS training partition)", fontsize=9.5); ax.grid(axis="x", alpha=0.25)
ax.set_xlim(0, float(imp.max())*1.15)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2E5B9A",label="Protocol-specified predictor"),Patch(color="#c0392b",label="Amendment addition")],
          fontsize=8.5, loc="lower right", framealpha=0.9)
ax.set_title("Global SHAP importance of the 17 predictors", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/shap_importance.png", dpi=140, bbox_inches="tight"); plt.close(fig)
print("   → figures/shap_importance.png (학습셋 기반)", flush=True)
print("\n저장: results/characteristics_availablecase.csv, bootstrap_validation.csv\n완료.", flush=True)
