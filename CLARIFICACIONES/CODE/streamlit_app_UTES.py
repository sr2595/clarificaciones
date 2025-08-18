import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO
from datetime import datetime
import unicodedata, re

st.set_page_config(page_title="Clarificador UTE", page_icon="📄", layout="wide")
st.title("📄 Clarificador UTE")

# --------- Helpers ---------
def _norm(texto):
    if texto is None:
        return ""
    s = str(texto)
    s = s.replace("\u00A0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def find_col(df, candidates):
    norm_map = { _norm(c): c for c in df.columns }
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    cand_norms = [_norm(c) for c in candidates]
    for orig in df.columns:
        n = _norm(orig)
        if any(cn in n or n in cn for cn in cand_norms if cn):
            return orig
    return None

def convertir_importe_europeo(valor):
    if pd.isna(valor):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip().replace('.', '').replace(',', '.')
    try:
        return float(texto)
    except Exception:
        return None

def es_ute(cif):
    if pd.isna(cif):
        return False
    cif = str(cif).upper().replace(" ", "").replace("-", "")
    if cif.startswith("L00") and len(cif) > 3:
        letra = cif[3]
        return letra == "U"
    return False

def limpiar_cif(cif):
    if pd.isna(cif):
        return ""
    return str(cif).upper().replace(" ", "").replace("-", "")

# --------- App ---------
archivo = st.file_uploader("Sube el archivo Excel", type=["xlsx", "xls"])
if archivo:
    try:
        df = pd.read_excel(archivo, engine="openpyxl")
    except Exception:
        df = pd.read_excel(archivo)

    with st.expander("🔎 Ver columnas detectadas en el Excel"):
        st.write(list(df.columns))

    # --- Detectar columnas ---
    col_fecha_emision = find_col(df, ['FECHA', 'Fecha Emision', 'Fecha Emisión', 'FX_EMISION'])
    col_factura       = find_col(df, ['FACTURA', 'Nº Factura', 'NRO_FACTURA', 'Núm.Doc.Deuda'])
    col_importe       = find_col(df, ['IMPORTE', 'TOTAL', 'TOTAL_FACTURA'])
    col_cif           = find_col(df, ['T.Doc. - Núm.Doc.', 'CIF', 'NIF', 'CIF_CLIENTE', 'NIF_CLIENTE'])
    col_nombre_cliente= find_col(df, ['NOMBRE', 'CLIENTE', 'RAZON_SOCIAL'])
    col_sociedad      = find_col(df, ['SOCIEDAD', 'Sociedad', 'SOC', 'EMPRESA'])
    col_grupo         = find_col(df, ['CIF_GRUPO', 'Grupo', 'CIF Grupo'])

    faltan = []
    if not col_fecha_emision: faltan.append("fecha emisión")
    if not col_factura:       faltan.append("nº factura")
    if not col_importe:       faltan.append("importe")
    if not col_cif:           faltan.append("CIF")
    if faltan:
        st.error("❌ No se pudieron localizar estas columnas: " + ", ".join(faltan))
        st.stop()

    # --- Normalizar ---
    df[col_fecha_emision] = pd.to_datetime(df[col_fecha_emision], dayfirst=True, errors='coerce')
    df[col_factura] = df[col_factura].astype(str)
    df[col_cif] = df[col_cif].astype(str)
    df['IMPORTE_CORRECTO'] = df[col_importe].apply(convertir_importe_europeo)
    df['IMPORTE_CENT'] = (df['IMPORTE_CORRECTO'] * 100).round().astype("Int64")
    df['CIF_LIMPIO'] = df[col_cif].apply(limpiar_cif)
    df['ES_UTE'] = df[col_cif].apply(es_ute)

    # --- Opciones de clientes finales (no UTES) ---
    df_clientes_unicos = df[~df['ES_UTE']][[col_cif, col_nombre_cliente, 'CIF_LIMPIO', col_grupo]].drop_duplicates()
    df_clientes_unicos[col_nombre_cliente] = df_clientes_unicos[col_nombre_cliente].fillna("").str.strip()
    df_clientes_unicos[col_cif] = df_clientes_unicos[col_cif].fillna("").str.strip()

    opciones_clientes = [
        f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
        for _, row in df_clientes_unicos.iterrows()
    ]
    mapping_cif = dict(zip(opciones_clientes, df_clientes_unicos[col_cif]))
    mapping_grupo = dict(zip(df_clientes_unicos[col_cif], df_clientes_unicos[col_grupo]))

    cliente_final_display = st.selectbox("Selecciona cliente final (CIF - Nombre)", opciones_clientes)
    cliente_final_cif = mapping_cif[cliente_final_display]
    cliente_final_grupo = mapping_grupo[cliente_final_cif]

    df_cliente_final = df[df[col_cif] == cliente_final_cif].copy()

    # --- Filtrar solo facturas TSS ---
    df_tss = df_cliente_final[df_cliente_final[col_sociedad] == 'TSS']
    if df_tss.empty:
        st.warning("❌ No hay facturas de TSS para este cliente final")
        st.stop()

    # --- Selección de factura final ---
    facturas_cliente = df_tss[[col_factura, col_fecha_emision, 'IMPORTE_CORRECTO']].dropna()
    opciones_facturas = [
        f"{row[col_factura]} - {row[col_fecha_emision].date()} - {row['IMPORTE_CORRECTO']:,.2f} €"
        for _, row in facturas_cliente.iterrows()
    ]
    factura_final_display = st.selectbox("Selecciona factura final TSS (90)", opciones_facturas)
    factura_final_id = factura_final_display.split(" - ")[0]
    factura_final = df_tss[df_tss[col_factura] == factura_final_id].iloc[0]
    st.info(f"Factura final seleccionada: **{factura_final[col_factura]}** ({factura_final['IMPORTE_CORRECTO']:,.2f} €)")

    # --- Filtrar UTES del mismo grupo ---
    df_utes_grupo = df[
        (df[col_grupo] == cliente_final_grupo) &
        (df['ES_UTE'])
    ].copy()

    opciones_utes = [
        f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
        for _, row in df_utes_grupo[[col_cif, col_nombre_cliente]].drop_duplicates().iterrows()
    ]
    mapping_utes_cif = dict(zip(opciones_utes, df_utes_grupo[col_cif]))

    socios_display = st.multiselect("Selecciona CIF(s) de la UTE (socios)", opciones_utes)
    socios_cifs = [mapping_utes_cif[s] for s in socios_display]
    df_internas = df[df[col_cif].isin(socios_cifs)].copy()

    if df_internas.empty:
        st.warning("❌ No hay facturas de UTE para los socios seleccionados")
        st.stop()

    # --- Solver con tolerancia ±1€ ---
    def cuadrar_internas(externa, df_internas):
        objetivo = int(externa['IMPORTE_CORRECTO']*100)
        data = list(zip(df_internas.index.tolist(),
                        df_internas['IMPORTE_CENT'].astype(int).tolist()))

        n = len(data)
        if n == 0:
            return None

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"sel_{i}") for i in range(n)]

        # Tolerancia ±1€
        tol = 100
        model.Add(sum(x[i]*data[i][1] for i in range(n)) >= objetivo - tol)
        model.Add(sum(x[i]*data[i][1] for i in range(n)) <= objetivo + tol)

        # Minimizar número de facturas
        model.Minimize(sum(x))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            indices_seleccionados = [data[i][0] for i in range(n) if solver.BooleanValue(x[i])]
            return df_internas.loc[indices_seleccionados]
        return None

    df_solucion = cuadrar_internas(factura_final, df_internas)
    if df_solucion is not None:
        st.success(f"✅ Se han encontrado {len(df_solucion)} facturas que cuadran con ±1€ de tolerancia")
        st.dataframe(df_solucion[[col_cif, col_nombre_cliente, col_factura, 'IMPORTE_CORRECTO', col_fecha_emision]])
    else:
        st.error("❌ No se pudo cuadrar la factura final con las facturas de socios dentro de ±1€")
