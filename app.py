# ================================================================
#   DASHBOARD EFH 2021 – Público (MISMO ESTILO)
#   Fuente: public_results/ (sin EFH cruda)
# ================================================================

import streamlit as st
st.set_page_config(layout="wide")  # debe ir primero

import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# ---------------------------------------------------------------
#   PALETA WOW
# ---------------------------------------------------------------
PRIMARY = "#4A90E2"
PURPLE = "#9B59B6"
TEAL = "#1ABC9C"
YELLOW = "#F1C40F"
DARK = "#2C3E50"
LIGHT = "#ECF0F1"

# ---------------------------------------------------------------
#   CSS WOW
# ---------------------------------------------------------------
st.markdown(f"""
    <style>

        .main {{
            background-color: {LIGHT};
        }}

        h1 {{
            text-align: center;
            background: linear-gradient(90deg, {PRIMARY}, {PURPLE});
            color: white !important;
            padding: 18px;
            border-radius: 12px;
            font-size: 40px !important;
            margin-bottom: 25px !important;
        }}

        h2 {{
            color: {DARK} !important;
            border-left: 10px solid {PRIMARY};
            padding-left: 12px;
            font-weight: 900 !important;
            margin-top: 15px !important;
            margin-bottom: 18px !important;
        }}

        .metric h3 {{
        font-size: 25px !important;
        }}

        [data-testid="stSidebar"] > div:first-child {{
            background-color: #1B2B65 !important;
            padding: 20px;
            border-radius: 10px;
        }}

        .metric {{
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 2px solid {PRIMARY};
            box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
#   CARGA DE RESULTADOS (public_results)
# ---------------------------------------------------------------
@st.cache_data
def load_public_eda():
    summary = pd.read_csv("eda_summary.csv")
    meta = pd.read_csv("eda_meta.csv")
    num_stats = pd.read_csv("eda_num_stats.csv")
    num_hist = pd.read_csv("eda_num_hist.csv.gz", compression="gzip")
    cat_counts = pd.read_csv("eda_cat_counts.csv.gz", compression="gzip")
    biv_num = pd.read_csv("eda_biv_num.csv")
    biv_cat = pd.read_csv("eda_biv_cat.csv.gz", compression="gzip")
    corr_long = pd.read_csv("eda_corr.csv.gz", compression="gzip")
    return summary, meta, num_stats, num_hist, cat_counts, biv_num, biv_cat, corr_long

@st.cache_data
def load_metrics_by_fold(path="metrics_by_fold.csv"):
    return pd.read_csv(path)

# ✅ Defaults (para que meta exista siempre)
summary = pd.DataFrame({"n_registros":[0], "pct_viv_propia":[0.0]})
meta = pd.DataFrame(columns=["var","type"])
num_stats = pd.DataFrame()
num_hist = pd.DataFrame()
cat_counts = pd.DataFrame()
biv_num = pd.DataFrame()
biv_cat = pd.DataFrame()
corr_long = pd.DataFrame()

try:
    summary, meta, num_stats, num_hist, cat_counts, biv_num, biv_cat, corr_long = load_public_eda()
except Exception as e:
    st.warning("No pude cargar EDA desde public_results/. El tablero seguirá solo con la calculadora.")
    st.caption(str(e))


# ---------------------------------------------------------------
# NORMALIZACIÓN DE COLUMNAS (compatibilidad public_results)
# ---------------------------------------------------------------
def _strip_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _rename_first_match(df, target, candidates):
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: target})
    return df

def normalize_public_tables():
    global summary, meta, num_stats, num_hist, cat_counts, biv_num, biv_cat, corr_long

    # limpiar espacios
    for name in ["summary","meta","num_stats","num_hist","cat_counts","biv_num","biv_cat","corr_long"]:
        df = globals().get(name, None)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            globals()[name] = _strip_cols(df)

    # ---- meta ----
    if meta is not None and isinstance(meta, pd.DataFrame) and not meta.empty:
        meta = _rename_first_match(meta, "var",  ["variable","Variable","VAR","feature","col","name"])
        meta = _rename_first_match(meta, "type", ["tipo","Tipo","TYPE","dtype","var_type"])
    else:
        meta = pd.DataFrame(columns=["var","type"])

    # ---- cat_counts ----
    if cat_counts is not None and isinstance(cat_counts, pd.DataFrame) and not cat_counts.empty:
        cat_counts = _rename_first_match(cat_counts, "var",      ["variable","Variable","VAR","feature","name"])
        cat_counts = _rename_first_match(cat_counts, "category", ["categoria","Categoría","cat","nivel","level","value"])
        cat_counts = _rename_first_match(cat_counts, "n",        ["count","Count","freq","frequency","N"])
        # macrozona: si no existe, crear "All"
        if "macrozona" not in cat_counts.columns:
            cat_counts["macrozona"] = "All"
    else:
        cat_counts = pd.DataFrame(columns=["macrozona","var","category","n"])

    # ---- num_hist ----
    if num_hist is not None and isinstance(num_hist, pd.DataFrame) and not num_hist.empty:
        num_hist = _rename_first_match(num_hist, "var",      ["variable","Variable","VAR","feature","name"])
        num_hist = _rename_first_match(num_hist, "bin_left", ["left","bin_l","lower","li","min"])
        num_hist = _rename_first_match(num_hist, "bin_right",["right","bin_r","upper","ls","max"])
        num_hist = _rename_first_match(num_hist, "count",    ["n","Count","freq","frequency","N"])
        if "macrozona" not in num_hist.columns:
            num_hist["macrozona"] = "All"
    else:
        num_hist = pd.DataFrame(columns=["macrozona","var","bin_left","bin_right","count"])

    # ---- num_stats ----
    if num_stats is not None and isinstance(num_stats, pd.DataFrame) and not num_stats.empty:
        num_stats = _rename_first_match(num_stats, "var", ["variable","Variable","VAR","feature","name"])
        if "macrozona" not in num_stats.columns:
            num_stats["macrozona"] = "All"
    else:
        num_stats = pd.DataFrame(columns=["macrozona","var"])

    # ---- biv_num / biv_cat ----
    for nm in ["biv_num","biv_cat"]:
        df = globals().get(nm, None)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            df = _rename_first_match(df, "var", ["variable","Variable","VAR","feature","name"])
            if "macrozona" not in df.columns:
                df["macrozona"] = "All"
            globals()[nm] = df

    # ---- corr_long ----
    if corr_long is not None and isinstance(corr_long, pd.DataFrame) and not corr_long.empty:
        corr_long = _rename_first_match(corr_long, "var1", ["x","v1","variable1","Variable1"])
        corr_long = _rename_first_match(corr_long, "var2", ["y","v2","variable2","Variable2"])
        corr_long = _rename_first_match(corr_long, "corr", ["cor","correlacion","correlation","value"])
    else:
        corr_long = pd.DataFrame(columns=["var1","var2","corr"])

    # 🔎 si aún falta algo esencial, mostrar columnas y detener
    if "var" not in cat_counts.columns or "category" not in cat_counts.columns:
        st.error("cat_counts no tiene las columnas esperadas después de normalizar.")
        st.write("cat_counts columns:", list(cat_counts.columns))
        st.stop()

normalize_public_tables()



try:
    df_metrics_fold = load_metrics_by_fold()
    have_fold_metrics = True
except Exception as e:
    have_fold_metrics = False
    st.warning("No pude cargar public_results/metrics_by_fold.csv (métricas por fold).")
    st.caption(str(e))

# Usamos All (sin filtro macrozona, igual que tu dashboard)
MACRO = "All"

# ---------------------------------------------------------------
#   COMPATIBILIDAD: si los CSV públicos no traen 'macrozona'
#   asumimos que todo corresponde a "All"
# ---------------------------------------------------------------
for df in (num_stats, num_hist, cat_counts, biv_num, biv_cat):
    if df is not None and "macrozona" not in df.columns:
        df["macrozona"] = "All"

# ---------------------------------------------------------------
#   DEFINICIÓN DE VARIABLES (las 18)
# ---------------------------------------------------------------
vars_rep = [
    # Numéricas
    'act_fijo', 'act_var', 'cap_pen_ent', 'edad_pr', 'hr_trabajadas_pr', 'yoprinm_pr', 'ypenh', 'ysubh',
    # Binarias
    't_cc', 't_tbco', 'u_cheq', 'u_pac', 'u_pat', 'u_tbco', 'u_prepago',
    # Categóricas
    'est_civil_pr', 'numh', 'ocuph',
]

available_vars = set(meta["var"].unique()) if (meta is not None and not meta.empty) else set(vars_rep)
vars_rep = [v for v in vars_rep if v in available_vars]



num_vars = meta.loc[meta["type"] == "numeric", "var"].tolist()
cat_vars = meta.loc[meta["type"] != "numeric", "var"].tolist()

# ---------------------------------------------------------------
#   SIDEBAR DE VARIABLES (igual)
# ---------------------------------------------------------------
st.sidebar.title("Variables 🔍")

search = st.sidebar.text_input("Buscar:", placeholder="Escriba para filtrar…")
filtered = [v for v in vars_rep if search.lower() in v.lower()] if search else vars_rep

if "var_selected" not in st.session_state:
    st.session_state["var_selected"] = filtered[0]

for v in filtered:
    if st.sidebar.button(v, key=f"btn_{v}"):
        st.session_state["var_selected"] = v

var = st.session_state["var_selected"]

# ---------------------------------------------------------------
#   TÍTULO PRINCIPAL
# ---------------------------------------------------------------
st.markdown("<h1>Dashboard Encuesta Financiera de Hogares 2021</h1>", unsafe_allow_html=True)

VAR_LABELS = {
    "yoprinm_pr": "ingreso mensual de la persona de referencia del hogar",
    "act_fijo": "monto total de activos financieros de renta fija",
    "act_var": "monto total que el hogar tiene invertido en los diferentes instrumentos de renta variable.",
    "cap_pen_ent": "saldo en cuenta de capitalización individual ",
    "edad_pr": "edad de la persona de referencia",
    "hr_trabajadas_pr": "horas trabajadas por la persona de referencia",
    "ypenh": "ingreso mensual del hogar por pensiones",
    "ysubh": "ingreso mensual del hogar por subsidios",
    "numh": "número de miembros en el hogar",
    "ocuph": "número de miembros del hogar que se encuentra trabajando",
    "u_pac": "si algún miembro del hogar utiliza pago automático a las cuentas corrientes",
    "t_tbco": "si algún miembro del hogar posee tarjetas de crédito bancarias",
    "u_pat": "si algún miembro del hogar utiliza pago automático a tarjetas de crédito",
    "u_tbco": "si algún miembro del hogar utiliza como medio de pago las tarjetas de crédito bancarias",
    "u_cheq": "si algún miembro del hogar utiliza como medio de pago los cheques",
    "t_cc": "si algún miembro del hogar posee cuenta corriente",
    "u_prepago": "si algún miembro del hogar utiliza como medio de pago los instrumentos de prepago",
    "est_civil_pr": "estado civil de la persona de referencia",
    
}

from pathlib import Path

# ---------------------------------------------------------------
#   SECCIÓN 0 — CALCULADORA DE PROBABILIDAD
# ---------------------------------------------------------------
st.header("🧮 Calculadora de probabilidad de vivienda propia")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "public_results" / "model_lr_18.joblib"

@st.cache_resource(show_spinner=False)
def load_mlp_model(model_path: Path):
    return joblib.load(model_path)

model_mlp = None
try:
    model_mlp = load_mlp_model(MODEL_PATH)
except Exception as e:
    st.error(f"No pude cargar el modelo en: {MODEL_PATH}")
    st.caption(str(e))
    st.stop()

# ✅ Safety extra (por si Streamlit re-ejecuta raro)
if model_mlp is None:
    st.error("El modelo no quedó cargado (model_mlp=None). Revisa el archivo joblib.")
    st.stop()

solo_calculadora = st.toggle("Modo simple (solo calculadora)", value=True)
st.caption("En modo simple se muestra únicamente la calculadora para evitar dependencias de EDA.")
    

# Inputs (puedes ajustar rangos luego)
col1, col2, col3 = st.columns(3)


#st.subheader("Perfiles rápidos (ejemplos)")

#if st.button("Perfil A: baja bancarización"):
    #st.session_state.update({
        #"u_pac": False, "t_cc": False, "t_tbco": False, "u_tbco": False, "u_pat": False, "u_cheq": False, "u_prepago": False
    #})

#if st.button("Perfil B: alta bancarización"):
    #st.session_state.update({
        #"u_pac": True, "t_cc": True, "t_tbco": True, "u_tbco": True, "u_pat": True, "u_cheq": False, "u_prepago": True
    #})


with col1:
    edad_pr = st.slider(VAR_LABELS["edad_pr"], 18, 100, 40)
    numh = st.slider(VAR_LABELS["numh"], 1, 12, 3)
    ocuph = st.slider(VAR_LABELS["ocuph"], 0, int(numh), 1)  # default 1

with col2:
    yoprinm_pr = st.number_input(VAR_LABELS["yoprinm_pr"], min_value=0, value=700000, step=50000)
    ypenh = st.number_input(VAR_LABELS["ypenh"], min_value=0, value=0, step=50000)
    ysubh = st.number_input(VAR_LABELS["ysubh"], min_value=0, value=0, step=50000)

with col3:
    act_fijo = st.number_input(VAR_LABELS["act_fijo"], min_value=0, value=0, step=500000)
    act_var = st.number_input(VAR_LABELS["act_var"], min_value=0, value=0, step=500000)
    cap_pen_ent = st.number_input(VAR_LABELS["cap_pen_ent"], min_value=0, value=0, step=500000)
    hr_trabajadas_pr = st.slider(VAR_LABELS["hr_trabajadas_pr"], 0, 80, 45)

st.subheader("Instrumentos financieros (0/1)")
b1, b2, b3, b4 = st.columns(4)

with b1:
    u_pac = st.checkbox(VAR_LABELS["u_pac"], value=False, key= "u_pac")
    t_cc = st.checkbox(VAR_LABELS["t_cc"], value=False, key= "t_cc")

with b2:
    t_tbco = st.checkbox(VAR_LABELS["t_tbco"], value=False, key= "t_tbco")
    u_tbco = st.checkbox(VAR_LABELS["u_tbco"], value=False, key= "u_tbco")

with b3:
    u_pat = st.checkbox(VAR_LABELS["u_pat"], value=False, key= "u_pat")
    u_cheq = st.checkbox(VAR_LABELS["u_cheq"], value=False, key= "u_cheq")

with b4:
    u_prepago = st.checkbox(VAR_LABELS["u_prepago"], value=False, key= "u_prepago")

st.subheader("Estado civil")
est_civil_pr = st.selectbox(
    VAR_LABELS["est_civil_pr"],
    options=["Soltero(a)", "Casado(a)", "Conviviente o pareja", "Divorciado(a)", "Viudo(a)", "Separado(a)"]
)

def money_to_model(x):
    x = max(float(x), 0.0)
    return np.log1p(x)

modo_estricto = st.checkbox("Aplicar validación de coherencia (recomendado)", value=True)   


# --- Coherencia de inputs (advertencia + opción estricta) ---
adj_yop = yoprinm_pr
adj_hr = hr_trabajadas_pr

# Advertencia si hay incoherencia típica
if ocuph == 0 and (yoprinm_pr > 0 or hr_trabajadas_pr > 0):
    st.warning("Advertencia: Ocupados=0 pero hay ingreso por actividad principal y/o horas trabajadas. "
               "Puede ser un caso atípico y afectar la estimación.")

# Si activas modo estricto, se fuerza coherencia laboral
if modo_estricto and ocuph == 0:
    adj_yop = 0.0
    adj_hr = 0

# Si horas=0 y aún hay ingreso principal, también es inconsistente (opcional)
if modo_estricto and adj_hr == 0 and adj_yop > 0:
    adj_yop = 0.0


# Construir fila de entrada con las 18 variables exactas
x_in = pd.DataFrame([{
    "act_fijo": money_to_model(act_fijo),
    "act_var": money_to_model(act_var),
    "cap_pen_ent": money_to_model(cap_pen_ent),
    "yoprinm_pr": money_to_model(adj_yop),
    "ypenh": money_to_model(ypenh),
    "ysubh": money_to_model(ysubh),

    # estas se dejan tal cual (no son montos monetarios)
    "edad_pr": edad_pr,
    "hr_trabajadas_pr": adj_hr,
    "numh": numh,
    "ocuph": ocuph,

    # binarias
    "t_cc": int(t_cc),
    "t_tbco": int(t_tbco),
    "u_cheq": int(u_cheq),
    "u_pac": int(u_pac),
    "u_pat": int(u_pat),
    "u_tbco": int(u_tbco),
    "u_prepago": int(u_prepago),

    # categórica
    "est_civil_pr": est_civil_pr,
}])

p = float(model_mlp.predict_proba(x_in)[0, 1])
st.metric("Probabilidad estimada de vivienda propia", f"{100*p:.2f}%")
st.caption(f"p = {p:.6f}")

st.progress(min(max(p, 0.0), 1.0))

if p < 0.33:
    st.info("Interpretación: probabilidad baja (según el modelo).")
elif p < 0.66:
    st.warning("Interpretación: probabilidad media (según el modelo).")
else:
    st.success("Interpretación: probabilidad alta (según el modelo).")

st.caption("Nota: la aplicación no carga microdatos EFH. Utiliza resultados agregados en public_results/ "
           "y un modelo entrenado para calcular la probabilidad.")


if solo_calculadora:
    st.stop()


# ---------------------------------------------------------------
#   SECCIÓN 1 — MÉTRICAS (igual, pero desde eda_summary)
# ---------------------------------------------------------------
st.header("📌 1. Resumen General")

n_reg = int(summary.loc[0, "n_registros"])
pct_vp = float(summary.loc[0, "pct_viv_propia"])

TOTAL_VARS = 18
NUMERICAS = 8
CATEGORICAS = 3
BINARIAS = 7

row1 = st.columns(6)
row1[0].markdown(f"<div class='metric'><h3>Registros</h3><h2>{n_reg:,}</h2></div>", unsafe_allow_html=True)
row1[1].markdown(f"<div class='metric'><h3>Vivienda Propia (%)</h3><h2>{pct_vp:.2f}%</h2></div>", unsafe_allow_html=True)
row1[2].markdown(f"<div class='metric'><h3>Total de Variables</h3><h2>{TOTAL_VARS}</h2></div>", unsafe_allow_html=True)
row1[3].markdown(f"<div class='metric'><h3>Variables Numéricas</h3><h2>{NUMERICAS}</h2></div>", unsafe_allow_html=True)
row1[4].markdown(f"<div class='metric'><h3>Variables Categóricas</h3><h2>{CATEGORICAS}</h2></div>", unsafe_allow_html=True)
row1[5].markdown(f"<div class='metric'><h3>Variables Binarias</h3><h2>{BINARIAS}</h2></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
#   helpers: reconstruir "df" sintético para gráficas sin EFH cruda
# ---------------------------------------------------------------
@st.cache_data
def pseudo_series_from_hist(var_name: str) -> pd.Series:
    h = num_hist[(num_hist["macrozona"] == MACRO) & (num_hist["var"] == var_name)].copy()
    if h.empty:
        return pd.Series([], dtype=float)
    h["bin_mid"] = (h["bin_left"] + h["bin_right"]) / 2.0
    vals = np.repeat(h["bin_mid"].to_numpy(), h["count"].to_numpy().astype(int))
    return pd.Series(vals, name=var_name)

@st.cache_data
def pseudo_biv_numeric(var_name: str, n_each: int = 2000) -> pd.DataFrame:
    # genera datos sintéticos por grupo usando cuantiles Q25/Q50/Q75
    row = biv_num[(biv_num["macrozona"] == MACRO) & (biv_num["var"] == var_name)]
    if row.empty:
        return pd.DataFrame({"viv_propia": [], var_name: []})
    row = row.iloc[0]

    def gen(q25, q50, q75, n, seed):
        rng = np.random.default_rng(seed)
        if any(pd.isna([q25, q50, q75])) or (q25 == q75):
            return np.full(n, q50 if not pd.isna(q50) else 0.0)
        u = rng.random(n)
        x = np.empty(n)
        left = u < 0.5
        x[left] = q25 + (q50 - q25) * (u[left] / 0.5)
        x[~left] = q50 + (q75 - q50) * ((u[~left] - 0.5) / 0.5)
        # muchos montos son >=0
        return np.clip(x, 0, None)

    seed0 = abs(hash(var_name)) % (2**32)
    x0 = gen(row["q25_0"], row["q50_0"], row["q75_0"], n_each, seed0 + 0)
    x1 = gen(row["q25_1"], row["q50_1"], row["q75_1"], n_each, seed0 + 1)

    dfb = pd.DataFrame({
        "viv_propia": np.r_[np.zeros(n_each, dtype=int), np.ones(n_each, dtype=int)],
        var_name: np.r_[x0, x1]
    })
    return dfb

# ---------------------------------------------------------------
#   SECCIÓN 2 — UNIVARIADO (igual en estructura)
# ---------------------------------------------------------------
st.header("📊 2. Análisis Univariadas")


def label_var(v: str) -> str:
    return VAR_LABELS.get(v, v)  # si no está, muestra el nombre

if var in num_vars and var not in ["numh", "ocuph"]:

    st.markdown("### Distribución")
    colD, colE = st.columns([2,2])

    srow = num_stats[(num_stats["macrozona"] == MACRO) & (num_stats["var"] == var)].iloc[0]
    total = int(srow["n"])
    cero = int(srow["cero"])
    pos = int(srow["pos"])
    porc_cero = float(srow["porc_cero"])
    porc_pos = float(srow["porc_pos"])

    with colD:
        st.subheader(" ")
        sig = label_var(var)
        cat0 = f"Hogares sin monto en {sig}"
        cat1 = f"Hogares con monto en {sig}"
        donut_df = pd.DataFrame({"categoria": [cat0, cat1], "valor": [cero, pos]})

        fig_donut = px.pie(
            donut_df, names="categoria", values="valor", hole=0.55,
            color="categoria",
            color_discrete_map={cat0: PRIMARY, cat1: TEAL}
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent", textfont_size=22)
        st.plotly_chart(fig_donut, use_container_width=True)

    with colE:
        st.subheader("🧠 Interpretación")
        st.markdown(f"""
        **Distribución general de `{sig}`:**
        - **{porc_cero:.2f}%** de los hogares no presentan **{sig}**.
        - **{porc_pos:.2f}%** tienen **{sig}**.

        """)
        

    st.markdown("---")
    colA, colB, colC = st.columns([2, 2, 1])

    serie = pseudo_series_from_hist(var)

    with colA:
        st.subheader("Histograma")
        fig = px.histogram(serie.to_frame(), x=var, nbins=30, color_discrete_sequence=[PRIMARY])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("BoxPlot")
        fig = px.box(serie.to_frame(), y=var, color_discrete_sequence=[TEAL])
        st.plotly_chart(fig, use_container_width=True)

    with colC:
        st.subheader("Resumen")
        st.write(serie.describe(percentiles=[0.25, 0.5, 0.75]))

else:
    colA, colC = st.columns([3, 1.5])

    raw = cat_counts[(cat_counts["macrozona"] == MACRO) & (cat_counts["var"] == var)].copy()

    # ✅ 1) Normalizar categorías numéricas: "2.0" -> "2"
    raw["category"] = raw["category"].astype(str).str.strip()
    raw["category"] = raw["category"].str.replace(r"\.0$", "", regex=True)

    # ✅ 2) Agrupar (por si venían duplicadas)
    raw = raw.groupby("category", as_index=False)["n"].sum()

    # ✅ 3) (Opcional recomendado) ordenar numéricamente si aplica
    raw["category_num"] = pd.to_numeric(raw["category"], errors="coerce")
    raw = raw.sort_values(["category_num", "n"], ascending=[True, False]).drop(columns=["category_num"])

    # ✅ 4) porcentaje
    # ✅ asegurar que "n" sea numérico
    raw["n"] = raw["n"].astype(str).str.replace(r"[^\d\.-]", "", regex=True)  # quita comas, %, espacios, etc.
    raw["n"] = pd.to_numeric(raw["n"], errors="coerce").fillna(0)

    # (opcional) si quieres n como entero
    raw["n"] = raw["n"].astype(int)

    # ✅ porcentaje seguro (evita división por 0)
    den = raw["n"].sum()
    if den == 0:
        raw["porcentaje"] = "0%"
    else:
        raw["porcentaje"] = (raw["n"] / den * 100).round(2).astype(str) + "%"

    # ✅ tabla final (solo 3 columnas)
    table = raw[["category", "n", "porcentaje"]].copy()
    table.columns = [var, "n", "porcentaje"]

    with colA:
        st.subheader("Categorías")
        fig = px.bar(table.head(25), x=var, y="n", text="porcentaje", color=var)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with colC:
        st.subheader("Frecuencias")
        st.dataframe(table.head(25), use_container_width=True)


# ---------------------------------------------------------------
#   SECCIÓN 3 — CORRELACIONES (desde eda_corr)
# ---------------------------------------------------------------
st.header("📈 3. Matriz de Correlación")

corr_mat = corr_long.pivot(index="var1", columns="var2", values="corr")
fig = px.imshow(corr_mat, text_auto=True, color_continuous_scale="Blues")
fig.update_layout(width=900, height=900)
st.plotly_chart(fig)

# ---------------------------------------------------------------
#   MODELOS (igual al piloto que ya te funcionó)
# ---------------------------------------------------------------
st.subheader("📌 Métricas por fold (LR Ridge vs MLP seleccionadas)")

if have_fold_metrics:
    metric_cols = ["Precision", "Sensibilidad", "Especificidad", "AUC"]
    metric_cols = [m for m in metric_cols if m in df_metrics_fold.columns]

    if len(metric_cols) == 0:
        st.error("No encuentro columnas de métricas en metrics_by_fold.csv.")
    else:
        dfp = df_metrics_fold[df_metrics_fold["Modelo"].str.contains("LR Ridge|MLP seleccionadas", case=False, regex=True)].copy()
        dfp["Fold"] = dfp["Fold"].astype(int)

        def plot_fold(metric_name, title):
            piv = dfp.pivot(index="Fold", columns="Modelo", values=metric_name).reset_index()
            long = piv.melt(id_vars="Fold", var_name="Modelo", value_name=metric_name)
            fig = px.line(long, x="Fold", y=metric_name, color="Modelo", markers=True, title=title)
            fig.update_layout(xaxis_title="Fold", yaxis_title=metric_name)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "Precision" in metric_cols:
                plot_fold("Precision", "Distribución por fold - Precisión")
            if "Sensibilidad" in metric_cols:
                plot_fold("Sensibilidad", "Distribución por fold - Sensibilidad")

        with col2:
            if "Especificidad" in metric_cols:
                plot_fold("Especificidad", "Distribución por fold - Especificidad")
            if "AUC" in metric_cols:
                plot_fold("AUC", "Distribución por fold - AUC")

    st.subheader("✅ Conclusión automática (criterio: promedio por fold)")
    metrics_use = ["AUC", "Sensibilidad", "Especificidad", "Precision"]
    metrics_use = [m for m in metrics_use if m in df_metrics_fold.columns]

    summary_models = df_metrics_fold.groupby("Modelo")[metrics_use].agg(["mean", "std"])
    st.dataframe(summary_models, use_container_width=True)

    best_model = df_metrics_fold.groupby("Modelo")["AUC"].mean().idxmax()
    st.success(f"📌 Según el **AUC promedio (OOF)**, el modelo recomendado es: **{best_model}**.")
else:
    st.info("No hay métricas por fold cargadas. (Falta public_results/metrics_by_fold.csv)")

