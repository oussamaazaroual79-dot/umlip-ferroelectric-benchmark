"""Build tidy benchmark table, scores and figures for the MLIP-ferroelectric paper."""
import glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

META = {  # label, family, training data
 "small": ("MACE-MP-0 small", "MACE", "MPtrj"), "medium": ("MACE-MP-0 medium", "MACE", "MPtrj"),
 "large": ("MACE-MP-0 large", "MACE", "MPtrj"), "medium-mpa-0": ("MACE-MPA-0", "MACE", "MPtrj+sAlex"),
 "medium-omat-0": ("MACE-OMAT-0", "MACE", "OMat24"), "mace-matpes-pbe-0": ("MACE-MatPES-PBE-0", "MACE", "MatPES-PBE"),
 "mace-matpes-r2scan-0": ("MACE-MatPES-r2SCAN-0", "MACE", "MatPES-r2SCAN"),
 "chgnet-0.3.0": ("CHGNet 0.3.0", "CHGNet", "MPtrj"), "7net-0": ("SevenNet-0", "SevenNet", "MPtrj"),
 "7net-mf-ompa": ("SevenNet-MF-ompa", "SevenNet", "OMat24+MPtrj+sAlex"),
 "MatterSim-v1.0.0-1M": ("MatterSim-v1 1M", "MatterSim", "MatterSim set"),
 "MatterSim-v1.0.0-5M": ("MatterSim-v1 5M", "MatterSim", "MatterSim set"),
 "orb-v3-conservative-inf-mpa": ("ORB-v3 (MPA)", "ORB", "MPtrj+Alex"),
 "orb-v3-conservative-inf-omat": ("ORB-v3 (OMat)", "ORB", "OMat24")}
FAMC = {"MACE": "#1f77b4", "CHGNet": "#d62728", "SevenNet": "#2ca02c", "MatterSim": "#9467bd", "ORB": "#ff7f0e"}
PB = {"P4mm": "T", "Amm2": "O", "R3m": "R"}

# ---------- perovskites ----------
pv = pd.concat([pd.read_csv(f) for f in ["results_step2_mace_all.csv"] + sorted(glob.glob("results_step4_*.csv"))], ignore_index=True)
dft = pd.read_csv("dft_results.csv"); dojo = pd.read_csv("dft_pseudo_check.csv")
rows = []
for _, r in pv[pv.phase != "Pm-3m"].iterrows():
    ref = dft[(dft.material == r.material) & (dft.phase == r.phase)].iloc[0]
    rows.append(dict(model=r.model, cls="perovskite", material=r.material, phase=r.phase, dE_ml=r.dE_meV_per_fu, dE_dft=ref.dE_meV_per_fu,
                     ca_ml=r.c_over_a, ca_dft=ref.c_over_a))
# ---------- fluorite oxides ----------
ox = pd.concat([pd.read_csv(f) for f in glob.glob("results_step5_*.csv") if "mp_ref" not in f], ignore_index=True)
odft = pd.concat([pd.read_csv(f"dft_oxides_{m}.csv") for m in ["HfO2", "ZrO2"]])
mp = pd.read_csv("results_step5_mp_reference.csv")
for _, r in ox[ox.phase != "P2_1/c"].iterrows():
    ref = odft[(odft.material == r.material) & (odft.phase == r.phase)].iloc[0]
    rows.append(dict(model=r.model, cls="fluorite", material=r.material, phase=r.phase, dE_ml=r.dE_meV_per_fu, dE_dft=ref.dE_meV_per_fu,
                     V_ml=r.vol_per_fu, V_dft=ref.vol_per_fu))
T = pd.DataFrame(rows); T["err"] = T.dE_ml - T.dE_dft
T["label"] = T.model.map(lambda m: META[m][0]); T["family"] = T.model.map(lambda m: META[m][1])
T.to_csv("benchmark_long.csv", index=False)

FE = ["BaTiO3", "PbTiO3", "KNbO3"]
def ground_ok(df_ml, df_ref, mats, phases):
    return sum(min(phases, key=lambda p: df_ml[(m, p)]) == min(phases, key=lambda p: df_ref[(m, p)]) for m in mats)
S = []
for m in META:
    t = T[T.model == m]
    pe = t[(t.cls == "perovskite") & t.material.isin(FE)]; fl = t[t.cls == "fluorite"]
    mlp = pv[pv.model == m].set_index(["material", "phase"]).dE_meV_per_fu.to_dict()
    refp = dft.set_index(["material", "phase"]).dE_meV_per_fu.to_dict()
    S.append(dict(model=m, label=META[m][0], family=META[m][1], data=META[m][2],
                  MAE_perov=pe.err.abs().mean(), MAE_HfO2=fl[fl.material == "HfO2"].err.abs().mean(),
                  MAE_ZrO2=fl[fl.material == "ZrO2"].err.abs().mean(),
                  depth_ratio=np.median(pe.dE_ml / pe.dE_dft), gs_perov=ground_ok(mlp, refp, FE, list(PB)),
                  STO_well=min(mlp[("SrTiO3", p)] for p in PB), KTO_well=min(mlp[("KTaO3", p)] for p in PB),
                  PTO_ca=pe[(pe.material == "PbTiO3") & (pe.phase == "P4mm")].ca_ml.iloc[0],
                  BTO_ca=pe[(pe.material == "BaTiO3") & (pe.phase == "P4mm")].ca_ml.iloc[0],
                  HfO2_Pca21=fl[(fl.material == "HfO2") & (fl.phase == "Pca2_1")].dE_ml.iloc[0],
                  ZrO2_Pca21=fl[(fl.material == "ZrO2") & (fl.phase == "Pca2_1")].dE_ml.iloc[0]))
S = pd.DataFrame(S); S["MAE_all"] = S[["MAE_perov", "MAE_HfO2", "MAE_ZrO2"]].mean(axis=1)
S = S.sort_values("MAE_all"); S.to_csv("scores_table.csv", index=False)
print(S.round(2).to_string(index=False))

# Spearman of ranking across classes (PBE-trained models only)
from scipy.stats import spearmanr
P = S[S.model != "mace-matpes-r2scan-0"]
R = {k: spearmanr(P[a], P[b]).statistic for k, (a, b) in
     {"perov_vs_HfO2": ("MAE_perov", "MAE_HfO2"), "perov_vs_ZrO2": ("MAE_perov", "MAE_ZrO2"), "HfO2_vs_ZrO2": ("MAE_HfO2", "MAE_ZrO2")}.items()}
# r2SCAN model vs MP r2SCAN energies
mpd = mp.set_index(["material", "phase"]).dE_mp_meV_per_fu
r2 = ox[(ox.model == "mace-matpes-r2scan-0") & (ox.phase != "P2_1/c")].set_index(["material", "phase"]).dE_meV_per_fu
R["r2scan_model_vs_MP_MAE"] = float((r2 - mpd.loc[r2.index]).abs().mean())
R["dojo_vs_gbrv"] = {m: dict(dE_gbrv=float(dft[(dft.material == m) & (dft.phase == "P4mm")].dE_meV_per_fu.iloc[0]),
                             dE_dojo=float(dojo[(dojo.material == m) & (dojo.phase == "P4mm")].dE_meV_per_fu.iloc[0]),
                             ca_gbrv=float(dft[(dft.material == m) & (dft.phase == "P4mm")].c_over_a.iloc[0]),
                             ca_dojo=float(dojo[(dojo.material == m) & (dojo.phase == "P4mm")].c_over_a.iloc[0])) for m in ["PbTiO3", "BaTiO3"]}
R["dft_perov"] = dft.set_index(["material", "phase"]).dE_meV_per_fu.round(2).rename(lambda x: x).to_dict()
R["dft_oxides"] = odft.set_index(["material", "phase"]).dE_meV_per_fu.round(2).to_dict()
R["mp_r2scan"] = mpd.round(2).to_dict()
json.dump({k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in R.items()},
          open("results_summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if not k.startswith(("dft", "mp"))}, indent=1, default=float))

# ---------- figures ----------
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
order = list(S.label)

# Fig 1: parity plot
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
for k, (a, sub) in enumerate([(ax[0], T[T.cls == "perovskite"]), (ax[1], T[T.cls == "fluorite"])]):
    for fam, g in sub.groupby("family"):
        g2 = g[g.model != "mace-matpes-r2scan-0"]
        a.scatter(g2.dE_dft, g2.dE_ml, s=14, alpha=0.75, color=FAMC[fam], label=fam, edgecolors="none")
    g = sub[sub.model == "mace-matpes-r2scan-0"]
    a.scatter(g.dE_dft, g.dE_ml, s=18, marker="x", color="k", label="MACE-MatPES-r2SCAN-0", lw=0.8)
    lo, hi = (sub[["dE_dft", "dE_ml"]].min().min() - 10, sub[["dE_dft", "dE_ml"]].max().max() + 10)
    a.plot([lo, hi], [lo, hi], "k-", lw=0.6); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel(r"DFT-PBE $\Delta E$ (meV/f.u.)"); a.set_ylabel(r"MLIP $\Delta E$ (meV/f.u.)")
    a.set_title(["(a) Perovskites: polar phases vs. cubic", "(b) HfO$_2$, ZrO$_2$: phases vs. monoclinic"][k], fontsize=9)
ax[1].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("paper/fig1_parity.pdf"); fig.savefig("paper/fig1_parity.png", dpi=200); plt.close()

# Fig 2: MAE per class
fig, a = plt.subplots(figsize=(7.2, 3.2)); x = np.arange(len(S)); w = 0.27
for i, (c, lab) in enumerate([("MAE_perov", "Perovskites"), ("MAE_HfO2", "HfO$_2$"), ("MAE_ZrO2", "ZrO$_2$")]):
    a.bar(x + (i - 1) * w, S[c], w, label=lab, color=["#4c72b0", "#dd8452", "#55a868"][i])
a.set_xticks(x); a.set_xticklabels([l + ("*" if "r2SCAN" in l else "") for l in S.label], rotation=55, ha="right", fontsize=7.5)
a.set_ylabel("MAE vs. DFT-PBE (meV/f.u.)"); a.legend(frameon=False, fontsize=8)
a.axhline(5, color="gray", lw=0.6, ls="--")
a.text(0.99, 0.97, "*r2SCAN-trained model, shown against the PBE reference", transform=a.transAxes, fontsize=6.5, ha="right", va="top", color="gray")
a.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("paper/fig2_mae.pdf"); fig.savefig("paper/fig2_mae.png", dpi=200); plt.close()

# Fig 3: c/a of PbTiO3 and BaTiO3
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
EXP = {"PbTiO3": 1.071, "BaTiO3": 1.010}   # as compiled in Zhang et al., PRB 96, 035143 (2017)
LIT = {"PbTiO3": (1.240, 1.122), "BaTiO3": (1.054, 1.029)}  # PBE (VASP-PAW) and SCAN, same reference
for a, mat, col in [(ax[0], "PbTiO3", "PTO_ca"), (ax[1], "BaTiO3", "BTO_ca")]:
    y = np.arange(len(S))
    a.scatter(S[col], y, c=[FAMC[f] for f in S.family], s=22, zorder=3)
    dv = R["dojo_vs_gbrv"][mat]
    a.axvspan(min(dv["ca_gbrv"], dv["ca_dojo"]), max(dv["ca_gbrv"], dv["ca_dojo"]), color="0.8", label="PBE, this work (GBRV/PseudoDojo)")
    a.axvline(EXP[mat], color="k", ls="--", lw=0.8, label="Experiment")
    a.axvline(LIT[mat][0], color="tab:red", ls=":", lw=1.0, label="PBE, PAW (Zhang et al.)")
    a.axvline(LIT[mat][1], color="tab:blue", ls="-.", lw=0.9, label="SCAN (Zhang et al.)")
    a.set_yticks(y); a.set_yticklabels(S.label if mat == "PbTiO3" else [], fontsize=7)
    a.set_xlabel("c/a (P4mm)"); a.set_title(f"({'ab'[mat=='BaTiO3']}) {mat.replace('3','$_3$')}", fontsize=9); a.invert_yaxis()
h, l = ax[1].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, frameon=False, loc="upper center", ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("paper/fig3_tetragonality.pdf"); fig.savefig("paper/fig3_tetragonality.png", dpi=200); plt.close()

# Fig 4: HfO2 / ZrO2 phase ladder
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)
PH = ["Pbca", "Pca2_1", "P4_2/nmc", "Fm-3m"]; PL = ["Pbca", "Pca2$_1$", "P4$_2$/nmc", "Fm$\\bar{3}$m"]
for a, mat in zip(ax, ["HfO2", "ZrO2"]):
    sub = T[T.material == mat]
    for j, ph in enumerate(PH):
        g = sub[sub.phase == ph]
        for _, r in g.iterrows():
            a.scatter(j + np.random.default_rng(hash(r.model) % 1000).uniform(-0.18, 0.18), r.dE_ml, s=12,
                      color="k" if r.model == "mace-matpes-r2scan-0" else FAMC[r.family], marker="x" if r.model == "mace-matpes-r2scan-0" else "o", alpha=0.8)
        a.hlines(g.dE_dft.iloc[0], j - 0.3, j + 0.3, color="r", lw=2, label="DFT-PBE (this work)" if j == 0 else None)
        mv = mp[(mp.material == mat) & (mp.phase == ph)].dE_mp_meV_per_fu.iloc[0]
        a.hlines(mv, j - 0.3, j + 0.3, color="gray", lw=1.2, ls=":", label="Materials Project r2SCAN" if j == 0 else None)
    a.set_xticks(range(4)); a.set_xticklabels(PL); a.set_ylabel(r"$E - E_{P2_1/c}$ (meV/f.u.)")
    a.set_title(f"({'ab'[mat=='ZrO2']}) {mat.replace('2','$_2$')}", fontsize=9)
ax[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("paper/fig4_fluorite.pdf"); fig.savefig("paper/fig4_fluorite.png", dpi=200); plt.close()

# Fig 5: signed-error heatmap
T["case"] = T.material.str.replace("2", "$_2$", regex=False).str.replace("3", "$_3$", regex=False) + " " + T.phase.map(lambda p: PB.get(p, p))
cases = list(dict.fromkeys(T[T.cls == "perovskite"].sort_values(["material", "phase"]).case)) + list(dict.fromkeys(T[T.cls == "fluorite"].sort_values(["material", "phase"]).case))
H = T.pivot_table(index="label", columns="case", values="err").loc[order, cases]
fig, a = plt.subplots(figsize=(7.4, 4.2)); im = a.imshow(H.values, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")
a.set_xticks(range(len(cases))); a.set_xticklabels([c.replace("_", "") for c in cases], rotation=70, ha="right", fontsize=6.5)
a.set_yticks(range(len(order))); a.set_yticklabels(order, fontsize=7)
cb = fig.colorbar(im, ax=a, fraction=0.03); cb.set_label(r"$\Delta E_{\rm MLIP}-\Delta E_{\rm DFT}$ (meV/f.u.)", fontsize=8)
fig.tight_layout(); fig.savefig("paper/fig5_heatmap.pdf"); fig.savefig("paper/fig5_heatmap.png", dpi=200); plt.close()
print("figures done")
