"""Fine-tuning (step 9) and molecular-dynamics (steps 7, 7b, 9) analysis: tables and figures 6-7."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
C = {"DFT": "#c0392b", "MACE-MPA-0": "#1f77b4", "MACE-MPA-0-FT": "#2ca02c", "ORB": "#ff7f0e"}

# ---------------- static benchmark of the fine-tuned model ----------------
dft = pd.read_csv("dft_results.csv")
st = pd.read_csv("ft/ft_static.csv")
m = st.merge(dft[["material", "phase", "dE_meV_per_fu", "c_over_a"]], on=["material", "phase"], suffixes=("", "_dft"))
m = m[m.phase != "Pm-3m"].copy(); m["err"] = m.dE_meV_per_fu - m.dE_meV_per_fu_dft
FE = ["BaTiO3", "PbTiO3", "KNbO3"]
out = {}
for mod, g in m.groupby("model"):
    out[mod] = dict(MAE_trained=g[g.in_training].err.abs().mean(), MAE_untrained=g[~g.in_training].err.abs().mean(),
                    MAE_untrained_KNO_STO=g[g.material.isin(["KNbO3", "SrTiO3"])].err.abs().mean(),
                    PTO_ca=g[(g.material == "PbTiO3") & (g.phase == "P4mm")].c_over_a.iloc[0],
                    BTO_ca=g[(g.material == "BaTiO3") & (g.phase == "P4mm")].c_over_a.iloc[0],
                    gs={mat: g[g.material == mat].sort_values("dE_meV_per_fu").phase.iloc[0] for mat in g.material.unique()})
err = pd.read_csv("ft/ft_errors.csv"); out["errors"] = err.round(3).to_dict(orient="records")

# ---------------- MD: phase identification from the polarisation components ----------------
def phase_of(r, pmin=0.10, comp=0.08):
    c = np.sort(np.abs([r.Px, r.Py, r.Pz]))[::-1]
    return "C" if c[0] < pmin else "TOR"[int((c > comp).sum()) - 1]
md_runs = {"ORB-v3 (MPA), run 1": ("md/md_orb_BaTiO3_50-600K.csv", "orb-v3-conservative-inf-mpa"),
           "ORB-v3 (MPA), run 2": ("md/md_orb_BaTiO3_50-600K_run2.csv", "orb-v3-conservative-inf-mpa"),
           "ORB-v3 (MPA), 650-1000 K": ("md/md_orb_step7b_PbTiO3_BaTiO3hot.csv", "orb-v3-conservative-inf-mpa"),
           "MACE-MPA-0": ("ft/md_ft.csv", "MACE-MPA-0"), "MACE-MPA-0-FT": ("ft/md_ft.csv", "MACE-MPA-0-FT")}
MD = {}
for k, (f, mod) in md_runs.items():
    d = pd.read_csv(f); d = d[d.model == mod]
    for mat, g in d.groupby("material"):
        g = g.sort_values("T_target").copy(); g["phase"] = g.apply(phase_of, axis=1); MD[(k, mat)] = g
def transitions(g):
    seq = list(zip(g.T_target, g.phase)); tr = {}
    for (t1, p1), (t2, p2) in zip(seq, seq[1:]):
        if p1 != p2: tr[f"{p1}->{p2}"] = (int(t1), int(t2))
    return tr
out["md_phases"] = {f"{k} | {mat}": " ".join(f"{t}:{p}" for t, p in zip(g.T_target, g.phase)) for (k, mat), g in MD.items()}
out["md_transitions_BaTiO3"] = {k: transitions(g) for (k, mat), g in MD.items() if mat == "BaTiO3" and "650" not in k}
json.dump(out, open("ft/ft_md_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

# ---------------- Fig 6: fine-tuning ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1, 1]))
PH = ["P4mm", "Amm2", "R3m"]; LAB = {"P4mm": "T", "Amm2": "O", "R3m": "R"}
for a, mats, title in [(ax[0], ["BaTiO3", "PbTiO3"], "(a) In the fine-tuning set"), (ax[1], ["KNbO3", "SrTiO3", "KTaO3"], "(b) Not in the fine-tuning set")]:
    keys = [(mt, ph) for mt in mats for ph in PH]; x = np.arange(len(keys)); w = 0.27
    ref = [dft[(dft.material == mt) & (dft.phase == ph)].dE_meV_per_fu.iloc[0] for mt, ph in keys]
    a.bar(x - w, ref, w, color=C["DFT"], label="DFT-PBE (this work)")
    for j, mod in enumerate(["MACE-MPA-0", "MACE-MPA-0-FT"]):
        v = [st[(st.model == mod) & (st.material == mt) & (st.phase == ph)].dE_meV_per_fu.iloc[0] for mt, ph in keys]
        a.bar(x + j * w, v, w, color=C[mod], label=mod.replace("-FT", " fine-tuned"))
    a.set_xticks(x); a.set_xticklabels([f"{mt.replace('3', '$_3$')}\n{LAB[ph]}" if ph == "Amm2" else LAB[ph] for mt, ph in keys], fontsize=7)
    a.axhline(0, color="k", lw=0.6); a.set_title(title, fontsize=9); a.set_ylabel(r"$\Delta E$ vs. cubic (meV/f.u.)")
ax[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig("paper/fig6_finetune.pdf"); fig.savefig("paper/fig6_finetune.png", dpi=200); plt.close()

# ---------------- Fig 7: MD ----------------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9))
sty = {"ORB-v3 (MPA), run 1": (C["ORB"], "o", "-"), "ORB-v3 (MPA), run 2": (C["ORB"], "s", "--"), "ORB-v3 (MPA), 650-1000 K": (C["ORB"], "^", ":"),
       "MACE-MPA-0": (C["MACE-MPA-0"], "o", "-"), "MACE-MPA-0-FT": (C["MACE-MPA-0-FT"], "o", "-")}
EXP = {"R-O": 183, "O-T": 278, "T-C": 393}
for k, (col, mk, ls) in sty.items():
    if (k, "BaTiO3") not in MD: continue
    g = MD[(k, "BaTiO3")]
    ax[0].plot(g.T_target, g.P_abs, marker=mk, ms=3, ls=ls, color=col, lw=1, label=(k.replace("-FT", " fine-tuned") if "650" not in k else None))
    ax[1].plot(g.T_target, g[["a", "b", "c"]].max(axis=1) / g[["a", "b", "c"]].min(axis=1), marker=mk, ms=3, ls=ls, color=col, lw=1)
for a in ax[:2]:
    for t in EXP.values(): a.axvline(t, color="gray", lw=0.6, ls=":")
ax[0].set_ylabel(r"$|\langle \mathbf{d}_{\rm Ti}\rangle|$ (\AA)".replace("\\AA", "Å")); ax[1].set_ylabel(r"$c_{\max}/a_{\min}$")
ax[0].set_title("(a) BaTiO$_3$ polarisation", fontsize=9); ax[1].set_title("(b) BaTiO$_3$ strain", fontsize=9)
for k, col in [("ORB-v3 (MPA), 650-1000 K", C["ORB"]), ("MACE-MPA-0-FT", C["MACE-MPA-0-FT"])]:
    if (k, "PbTiO3") in MD:
        g = MD[(k, "PbTiO3")]
        ax[2].plot(g.T_target, g.P_abs, marker="o", ms=3, color=col, lw=1, label="ORB-v3 (MPA)" if "ORB" in k else "MACE-MPA-0 fine-tuned")
ax[2].axvline(763, color="gray", lw=0.6, ls=":"); ax[2].set_ylim(0, 0.7)
ax[2].set_title("(c) PbTiO$_3$ polarisation", fontsize=9); ax[2].set_ylabel(r"$|\langle \mathbf{d}_{\rm Ti}\rangle|$ (Å)")
ax[2].legend(fontsize=6.5, frameon=False, loc="lower right")
for a in ax: a.set_xlabel("T (K)")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, frameon=False, loc="upper center", ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("paper/fig7_md.pdf"); fig.savefig("paper/fig7_md.png", dpi=200); plt.close()
print("figs 6-7 done")
