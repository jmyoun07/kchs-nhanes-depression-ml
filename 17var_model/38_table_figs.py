# -*- coding: utf-8 -*-
"""Table 3(특성)·Table 4(민감도)를 그림으로. 기존 그림 팔레트와 일치(KCHS 파랑/pre 빨강/post 초록)."""
import sys, numpy as np; sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/results/figures"
K,PRE,POST="#2E5B9A","#c0392b","#27ae60"

# ── Figure 6: 표본 특성 (Table 3) — 2 패널 ──
# 값은 available-case 산출물에서 읽는다(하드코딩 금지) — 53_review3_fixes.py 산출
import pandas as _pd
_ch=_pd.read_csv("C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/results/characteristics_availablecase.csv").set_index("characteristic")
_g=lambda k:(round(float(_ch.loc[k,"KCHS"]),2),round(float(_ch.loc[k,"NHANES pre"]),2),round(float(_ch.loc[k,"NHANES post"]),2))
pct=[("Depression (PHQ-9≥{}0)".format(1),)+_g("Depression PHQ-9 >=10"),("Female",)+_g("Female"),("Current smoker",)+_g("Current smoker"),
     ("Partnered",)+_g("Partnered"),("Employed",)+_g("Employed"),("Food insecure",)+_g("Food insecure"),
     ("Hypertension",)+_g("Hypertension"),("Diabetes",)+_g("Diabetes")]
cont=[("Age (years)",)+_g("Age (years)"),("BMI",)+_g("BMI"),("Sleep (hours)",)+_g("Sleep hours"),
      ("Self-rated health\n(1 best–5 worst)",)+_g("Self-rated health"),
      ("Oral health\n(1 best–5 worst)",)+_g("Oral health")]
fig,axes=plt.subplots(1,2,figsize=(11,5.2),gridspec_kw={"width_ratios":[1,0.8]})
def grouped(ax,data,title,xlabel,xlim=None):
    labels=[d[0] for d in data]; y=np.arange(len(data)); h=0.26
    ax.barh(y+h,[d[1] for d in data],h,color=K,label="KCHS")
    ax.barh(y,   [d[2] for d in data],h,color=PRE,label="NHANES pre")
    ax.barh(y-h,[d[3] for d in data],h,color=POST,label="NHANES post")
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8.5); ax.invert_yaxis()
    ax.set_title(title,fontsize=10,fontweight="bold"); ax.set_xlabel(xlabel,fontsize=9)
    if xlim: ax.set_xlim(*xlim)
    ax.grid(axis="x",alpha=0.25); ax.tick_params(labelsize=8)
grouped(axes[0],pct,"A. Prevalence / proportion","Weighted %")
grouped(axes[1],cont,"B. Means","Weighted mean")
axes[0].legend(fontsize=8.5,loc="lower right",framealpha=0.9)
fig.suptitle("Weighted sample characteristics across cohorts (available cases)",fontsize=11,fontweight="bold",y=1.0)
fig.tight_layout(); fig.savefig(f"{FIG}/characteristics.png",dpi=140,bbox_inches="tight"); plt.close(fig)
print("→ characteristics.png")

# ── Figure 7: 민감도 (Appendix 3 Table S7) — forest. 값은 결과 CSV에서 읽는다(하드코딩 금지) ──
import pandas as pd
RES="C:/Users/rudtn/Desktop/claude code/지역사회건강/17var_model/results"
_s=pd.read_csv(f"{RES}/final_sensitivity.csv").set_index("cond")
_f=pd.read_csv(f"{RES}/sensitivity_full.csv").set_index("cond")
def _row(df,key): r=df.loc[key]; return (round(float(r["internal"]),3),round(float(r["ext_pre"]),3),round(float(r["ext_post"]),3))
spec=[("Primary (17 predictors)",)+_row(_s,"기준(17변수)"),
      ("Food insecurity excluded (16)",)+_row(_s,"food_insecure 제외(16)"),
      ("Three added excluded (14, retuned)",)+_row(_s,"신규3 제외(14, 자체 튜닝)"),
      ("Weighted training",)+_row(_f,"가중 학습"),
      ("Missing-indicator imputation",)+_row(_f,"결측지표 추가"),
      ("Single chained-equations imputation",)+_row(_f,"MICE 대체"),
      ("Age ≥30 y (train + evaluate)",)+_row(_s,"연령≥30(학습·평가 모두 제한)"),
      ("Outcome cut-off ≥5",)+_row(_s,"컷오프≥5(전체 산출)")]
fig,ax=plt.subplots(figsize=(8.4,5.0))
y=np.arange(len(spec))[::-1]
for i,(lab,a,b,c) in zip(y,spec):
    ax.plot([b,c,a],[i,i,i],color="#ccc",lw=0.8,zorder=1)
ax.scatter([d[1] for d in spec],y,color=K,label="Internal",zorder=3,s=42)
ax.scatter([d[2] for d in spec],y,color=PRE,label="NHANES pre",zorder=3,s=42,marker="s")
ax.scatter([d[3] for d in spec],y,color=POST,label="NHANES post",zorder=3,s=42,marker="^")
ax.axvline(0.80,color="#888",ls=":",lw=0.9,alpha=0.7); ax.text(0.801,0.15,"0.80 target",fontsize=7.5,color="#666")
ax.set_yticks(y); ax.set_yticklabels([d[0] for d in spec],fontsize=8.5)
ax.set_xlabel("Weighted AUROC",fontsize=9.5); ax.set_xlim(0.70,0.86); ax.grid(axis="x",alpha=0.25)
ax.legend(fontsize=8.5,loc="upper left",framealpha=0.9)
ax.set_title("Sensitivity analyses — weighted AUROC across specifications",fontsize=10.5,fontweight="bold")
fig.tight_layout(); fig.savefig(f"{FIG}/sensitivity.png",dpi=140,bbox_inches="tight"); plt.close(fig)
print("→ sensitivity.png")
print("완료.")
