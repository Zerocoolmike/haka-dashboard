import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from itertools import combinations

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HAKA I lend – Dashboard", layout="wide")
st.title("HAKA I lend – 5 mõõtmiskorda")

import os
# Otsib .sav faili samast kaustast
_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_DIR, "HAKA_I_lend_5_korda_koos_04112025.sav")

SCHOOL_LABELS = {
    1: "Mahtra PK",
    2: "Merivälja Kool",
    3: "Kuristiku Gümn.",
    4: "Tallinna Ühisgümn.",
    5: "Viimsi Kool",
}

WAVE_NAMES = ["1. mõõtmine", "2. mõõtmine", "3. mõõtmine", "4. mõõtmine", "5. mõõtmine"]

# Construct definitions: (display_name, {wave_index: [columns]})
# Wave prefixes: 0="" 1="a" 2="b" 3="c" 4="d"
CONSTRUCTS = {
    "TL (Õpetaja eestvedamine)": {
        0: ["TL1", "TL2", "TL3", "TL4"],
        1: ["aTL1", "aTL2", "aTL3", "aTL4"],
        2: ["bTL1", "bTL2", "bTL3", "bTL4"],
        3: ["cTL1", "cTL2", "cTL3", "cTL4"],
        4: ["dTL1", "dTL2", "dTL3", "dTL4"],
    },
    "Andmete kasutamine": {
        0: ["data1", "data2", "data3", "data4"],
        1: ["adata1", "adata2", "adata3", "adata4"],
        2: ["bdata1", "bdata2", "bdata3", "bdata4"],
        3: ["cdata1", "cdata2", "cdata3", "cdata4"],
        4: ["ddata1", "ddata2", "ddata3", "ddata4"],
    },
    "Dialoog": {
        0: ["dialoog1", "dialoog2"],
        1: ["adialoog1", "adialoog2"],
        2: ["bdialoog1", "bdialoog2"],
        3: ["cdialoog1", "cdialoog2"],
        4: ["ddialoog1", "ddialoog2"],
    },
    "Tähendus": {
        0: ["tahendus1", "tahendus2"],
        1: ["atahendus1", "atahendus2"],
        2: ["btahendus1", "btahendus2"],
        3: ["ctahendus1", "ctahendus2"],
        4: ["dtahendus1", "dtahendus2"],
    },
    "Omand": {
        0: ["omand1", "omand2"],
        1: ["aomand1", "aomand2"],
        2: ["bomand1", "bomand2"],
        3: ["comand1", "comand2"],
        4: ["domand1", "domand2"],
    },
    "Eestvedamine": {
        0: ["eestvedav1", "eestvedav5"],
        1: ["aeestvedav1", "aeestvedav5"],
        2: ["beestvedav1", "beestvedav5"],
        3: ["ceestvedav1_teineskaala", "ceestvedav5_teineskaala"],
        4: ["deestvedav1", "deestvedav5"],
    },
}

SCHOOL_CODE_COLS = {
    0: "kooli_kood",
    1: "a_kooli_kood2",
    2: "b_kool",
    3: "c_koolikood",
    4: "d_koolikood",
}


# ── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df, meta = pyreadstat.read_sav(DATA_PATH)
    return df


df = load_data()


def compute_composite(df, cols):
    """Compute mean of cols, requiring at least half non-missing."""
    sub = df[cols]
    min_valid = max(1, len(cols) // 2)
    return sub.mean(axis=1).where(sub.notna().sum(axis=1) >= min_valid)


def build_long_data(df, construct_name):
    """Build a long-form DataFrame with columns: school, wave, score."""
    waves_def = CONSTRUCTS[construct_name]
    rows = []
    for wave_idx in range(5):
        cols = waves_def[wave_idx]
        existing = [c for c in cols if c in df.columns]
        if not existing:
            continue
        score = compute_composite(df, existing)
        school_col = SCHOOL_CODE_COLS[wave_idx]
        school = df[school_col]
        for i in range(len(df)):
            s = score.iloc[i]
            sch = school.iloc[i]
            if pd.notna(s) and pd.notna(sch):
                rows.append({
                    "school": SCHOOL_LABELS.get(int(sch), f"Kool {int(sch)}"),
                    "school_code": int(sch),
                    "wave": WAVE_NAMES[wave_idx],
                    "wave_idx": wave_idx,
                    "score": s,
                })
    return pd.DataFrame(rows)


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Seaded")
selected_construct = st.sidebar.selectbox("Vali konstrukt", list(CONSTRUCTS.keys()))
selected_schools = st.sidebar.multiselect(
    "Vali koolid",
    list(SCHOOL_LABELS.values()),
    default=list(SCHOOL_LABELS.values()),
)
alpha = st.sidebar.slider("Olulisuse nivoo (α)", 0.01, 0.10, 0.05, 0.01)

long_df = build_long_data(df, selected_construct)
long_df = long_df[long_df["school"].isin(selected_schools)]

# ── Tab layout ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Kirjeldav statistika",
    "Koolide võrdlus",
    "Muutused ajas (kooli sees)",
    "Üksikküsimused",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – Descriptive statistics
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Kirjeldav statistika")
    st.subheader(selected_construct)

    desc = (
        long_df.groupby(["school", "wave"])["score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(columns={
            "count": "N", "mean": "Keskmine", "median": "Mediaan",
            "std": "SD", "min": "Min", "max": "Max",
        })
        .reset_index()
        .rename(columns={"school": "Kool", "wave": "Mõõtmine"})
    )
    desc = desc.sort_values(["Kool", "Mõõtmine"])
    st.dataframe(
        desc.style.format({
            "Keskmine": "{:.2f}", "Mediaan": "{:.2f}",
            "SD": "{:.2f}", "Min": "{:.1f}", "Max": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Overall summary per school (across all waves)
    st.subheader("Koondstatistika koolide kaupa (kõik mõõtmised)")
    overall = (
        long_df.groupby("school")["score"]
        .agg(["count", "mean", "median", "std"])
        .rename(columns={
            "count": "N", "mean": "Keskmine", "median": "Mediaan", "std": "SD",
        })
        .reset_index()
        .rename(columns={"school": "Kool"})
    )
    st.dataframe(
        overall.style.format({"Keskmine": "{:.2f}", "Mediaan": "{:.2f}", "SD": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Box plots
    st.subheader("Jaotus mõõtmiskordade ja koolide kaupa")
    fig_box = px.box(
        long_df, x="wave", y="score", color="school",
        labels={"wave": "Mõõtmine", "score": "Skoor", "school": "Kool"},
        category_orders={"wave": WAVE_NAMES},
    )
    fig_box.update_layout(height=500)
    st.plotly_chart(fig_box, use_container_width=True)

    # Line chart of means
    st.subheader("Keskmiste muutus ajas")
    means = long_df.groupby(["school", "wave", "wave_idx"])["score"].mean().reset_index()
    means = means.sort_values("wave_idx")
    fig_line = px.line(
        means, x="wave", y="score", color="school", markers=True,
        labels={"wave": "Mõõtmine", "score": "Keskmine skoor", "school": "Kool"},
        category_orders={"wave": WAVE_NAMES},
    )
    fig_line.update_layout(height=450)
    st.plotly_chart(fig_line, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – Between-school comparison (Kruskal-Wallis + pairwise Mann-Whitney)
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Koolide võrdlus (Kruskal-Wallis + Mann-Whitney U)")
    st.subheader(selected_construct)

    for wave_name in WAVE_NAMES:
        wave_data = long_df[long_df["wave"] == wave_name]
        groups = [g["score"].values for _, g in wave_data.groupby("school")]
        school_names = [name for name, _ in wave_data.groupby("school")]

        if len(groups) < 2:
            continue

        st.markdown(f"### {wave_name}")

        # Kruskal-Wallis
        if len(groups) >= 2 and all(len(g) >= 1 for g in groups):
            kw_stat, kw_p = stats.kruskal(*groups)
            sig = "Statistiliselt oluline" if kw_p < alpha else "Ei ole statistiliselt oluline"
            st.markdown(f"**Kruskal-Wallis:** H = {kw_stat:.3f}, p = {kw_p:.4f} — {sig}")

            # Pairwise Mann-Whitney if significant
            if kw_p < alpha:
                st.markdown("**Paariviisiline võrdlus (Mann-Whitney U):**")
                pw_rows = []
                pairs = list(combinations(range(len(groups)), 2))
                # Bonferroni correction
                n_comparisons = len(pairs)
                for i, j in pairs:
                    if len(groups[i]) >= 1 and len(groups[j]) >= 1:
                        u_stat, u_p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                        adjusted_alpha = alpha / n_comparisons
                        sig_pw = "Jah" if u_p < adjusted_alpha else "Ei"
                        # Effect size r = Z / sqrt(N)
                        n_total = len(groups[i]) + len(groups[j])
                        z_val = stats.norm.ppf(1 - u_p / 2) if u_p < 1 else 0
                        r_effect = z_val / np.sqrt(n_total) if n_total > 0 else 0
                        pw_rows.append({
                            "Paar": f"{school_names[i]} vs {school_names[j]}",
                            "U": f"{u_stat:.1f}",
                            "p": f"{u_p:.4f}",
                            f"Oluline (a={adjusted_alpha:.4f})": sig_pw,
                            "Efekti suurus (r)": f"{r_effect:.3f}",
                        })
                if pw_rows:
                    st.dataframe(pd.DataFrame(pw_rows), use_container_width=True, hide_index=True)
                    st.caption(f"Bonferroni korrektsioon: a = {alpha} / {n_comparisons} = {alpha / n_comparisons:.4f}")
        st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – Within-school changes over time (Friedman + Wilcoxon)
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Muutused ajas kooli sees")
    st.subheader(selected_construct)
    st.markdown("""
    **Meetod:** Iga mõõtmispaari jaoks kasutatakse Wilcoxon signed-rank testi
    (mitteparameetriline paaride test). Testitakse, kas kahe järjestikuse
    mõõtmiskorra vahel on statistiliselt oluline erinevus.

    Lisaks kasutatakse Friedmani testi, kui on vähemalt 3 mõõtmiskorda,
    kus sama vastaja on osalenud.
    """)

    waves_def = CONSTRUCTS[selected_construct]

    for school_name in selected_schools:
        school_code = [k for k, v in SCHOOL_LABELS.items() if v == school_name]
        if not school_code:
            continue
        school_code = school_code[0]

        st.markdown(f"### {school_name}")

        # Compute per-wave composite scores for this school
        wave_scores = {}
        for wave_idx in range(5):
            cols = waves_def[wave_idx]
            existing = [c for c in cols if c in df.columns]
            if not existing:
                continue
            school_col = SCHOOL_CODE_COLS[wave_idx]
            mask = df[school_col] == school_code
            score = compute_composite(df.loc[mask], existing)
            wave_scores[wave_idx] = score

        # Pairwise Wilcoxon signed-rank tests between consecutive waves
        # We need matched participants - use row index as proxy
        st.markdown("**Paariviisiline Wilcoxon signed-rank test (järjestikused mõõtmised):**")
        pw_results = []
        wave_indices = sorted(wave_scores.keys())

        for idx_a, idx_b in combinations(wave_indices, 2):
            scores_a = wave_scores[idx_a]
            scores_b = wave_scores[idx_b]
            # Match on index (same row = same person in wide format)
            common_idx = scores_a.dropna().index.intersection(scores_b.dropna().index)
            if len(common_idx) < 5:
                pw_results.append({
                    "Paar": f"{WAVE_NAMES[idx_a]} vs {WAVE_NAMES[idx_b]}",
                    "N (paarid)": len(common_idx),
                    "Mediaan 1": "–",
                    "Mediaan 2": "–",
                    "W": "–",
                    "p": "–",
                    "Oluline": "Liiga vähe andmeid",
                    "Efekti suurus (r)": "–",
                })
                continue

            a_vals = scores_a.loc[common_idx].values
            b_vals = scores_b.loc[common_idx].values
            try:
                w_stat, w_p = stats.wilcoxon(a_vals, b_vals)
                n_pairs = len(common_idx)
                r_effect = abs(stats.norm.ppf(w_p / 2)) / np.sqrt(n_pairs) if w_p < 1 else 0
                # Bonferroni
                n_comp = len(list(combinations(wave_indices, 2)))
                adj_alpha = alpha / n_comp
                sig = "Jah" if w_p < adj_alpha else "Ei"
                pw_results.append({
                    "Paar": f"{WAVE_NAMES[idx_a]} vs {WAVE_NAMES[idx_b]}",
                    "N (paarid)": n_pairs,
                    "Mediaan 1": f"{np.median(a_vals):.2f}",
                    "Mediaan 2": f"{np.median(b_vals):.2f}",
                    "W": f"{w_stat:.1f}",
                    "p": f"{w_p:.4f}",
                    f"Oluline (a={adj_alpha:.4f})": sig,
                    "Efekti suurus (r)": f"{r_effect:.3f}",
                })
            except Exception:
                pw_results.append({
                    "Paar": f"{WAVE_NAMES[idx_a]} vs {WAVE_NAMES[idx_b]}",
                    "N (paarid)": len(common_idx),
                    "Mediaan 1": f"{np.median(a_vals):.2f}",
                    "Mediaan 2": f"{np.median(b_vals):.2f}",
                    "W": "–",
                    "p": "–",
                    "Oluline": "Ei saanud arvutada",
                    "Efekti suurus (r)": "–",
                })

        if pw_results:
            st.dataframe(pd.DataFrame(pw_results), use_container_width=True, hide_index=True)

        # Friedman test (needs matched data across 3+ waves)
        if len(wave_indices) >= 3:
            # Find common index across all available waves
            common = wave_scores[wave_indices[0]].dropna().index
            for wi in wave_indices[1:]:
                common = common.intersection(wave_scores[wi].dropna().index)

            if len(common) >= 5:
                friedman_data = [wave_scores[wi].loc[common].values for wi in wave_indices]
                try:
                    fr_stat, fr_p = stats.friedmanchisquare(*friedman_data)
                    sig = "Statistiliselt oluline" if fr_p < alpha else "Ei ole statistiliselt oluline"
                    st.markdown(
                        f"**Friedmani test** (kõik mõõtmised, N={len(common)} paaritatud vastajat): "
                        f"chi2 = {fr_stat:.3f}, p = {fr_p:.4f} — {sig}"
                    )
                except Exception as e:
                    st.warning(f"Friedmani testi ei saanud arvutada: {e}")
            else:
                st.info(f"Friedmani test: liiga vähe paaritatud vastajaid ({len(common)}) kõigi mõõtmiste lõikes.")

        # Violin plot for this school
        school_data = long_df[long_df["school"] == school_name]
        if len(school_data) > 0:
            fig_v = px.violin(
                school_data, x="wave", y="score", box=True, points="all",
                labels={"wave": "Mõõtmine", "score": "Skoor"},
                category_orders={"wave": WAVE_NAMES},
                title=f"{school_name} – skooride jaotus",
            )
            fig_v.update_layout(height=400)
            st.plotly_chart(fig_v, use_container_width=True)

        st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – Individual items
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Üksikküsimuste analüüs")
    st.subheader(selected_construct)

    waves_def = CONSTRUCTS[selected_construct]

    # Build item-level long data
    item_rows = []
    for wave_idx in range(5):
        cols = waves_def[wave_idx]
        school_col = SCHOOL_CODE_COLS[wave_idx]
        for col in cols:
            if col not in df.columns:
                continue
            # Derive clean item name (remove wave prefix)
            item_name = col
            for prefix in ["a", "b", "c", "d"]:
                if col.startswith(prefix) and not col.startswith("data") and not col.startswith("dialoog"):
                    item_name = col[len(prefix):]
                    break
            # Handle special cases
            if "teineskaala" in col:
                item_name = col.replace("ceestvedav", "eestvedav").replace("_teineskaala", "")

            for i in range(len(df)):
                val = df[col].iloc[i]
                sch = df[school_col].iloc[i]
                if pd.notna(val) and pd.notna(sch):
                    item_rows.append({
                        "item": item_name,
                        "school": SCHOOL_LABELS.get(int(sch), f"Kool {int(sch)}"),
                        "wave": WAVE_NAMES[wave_idx],
                        "wave_idx": wave_idx,
                        "score": val,
                    })

    item_df = pd.DataFrame(item_rows)
    item_df = item_df[item_df["school"].isin(selected_schools)]

    if len(item_df) > 0:
        unique_items = sorted(item_df["item"].unique())
        for item_name in unique_items:
            st.markdown(f"#### {item_name}")
            idata = item_df[item_df["item"] == item_name]

            # Summary table
            item_summary = (
                idata.groupby(["school", "wave"])["score"]
                .agg(["count", "mean", "median", "std"])
                .rename(columns={
                    "count": "N", "mean": "Keskmine",
                    "median": "Mediaan", "std": "SD",
                })
                .reset_index()
                .rename(columns={"school": "Kool", "wave": "Mõõtmine"})
            )
            st.dataframe(
                item_summary.style.format({
                    "Keskmine": "{:.2f}", "Mediaan": "{:.2f}", "SD": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Line chart per item
            imeans = idata.groupby(["school", "wave", "wave_idx"])["score"].mean().reset_index()
            imeans = imeans.sort_values("wave_idx")
            fig_item = px.line(
                imeans, x="wave", y="score", color="school", markers=True,
                labels={"wave": "Mõõtmine", "score": "Keskmine", "school": "Kool"},
                category_orders={"wave": WAVE_NAMES},
            )
            fig_item.update_layout(height=350)
            st.plotly_chart(fig_item, use_container_width=True)

    else:
        st.warning("Andmeid ei leitud valitud filtritega.")

# ── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("HAKA I lend – 5 mõõtmiskorda dashboard")
st.sidebar.caption(f"Andmestik: {len(df)} rida, {len(df.columns)} tunnust")
