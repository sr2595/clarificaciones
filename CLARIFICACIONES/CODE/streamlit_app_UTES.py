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

    # --- Selección de cliente final ---
    opciones_clientes = sorted(df[col_cif].dropna().unique())
    cliente_final_cif = st.selectbox("Selecciona cliente final (CIF)", opciones_clientes)
    df_cliente_final = df[df[col_cif] == cliente_final_cif].copy()

    # --- Selección de factura final (90) ---
    facturas_cliente = df_cliente_final[[col_factura, col_fecha_emision, 'IMPORTE_CORRECTO']].dropna()
    opciones_facturas = [
        f"{row[col_factura]} - {row[col_fecha_emision].date()} - {row['IMPORTE_CORRECTO']:,.2f} €"
        for _, row in facturas_cliente.iterrows()
    ]
    factura_final_display = st.selectbox("Selecciona factura final (90)", opciones_facturas)
    factura_final_id = factura_final_display.split(" - ")[0]
    factura_final = df_cliente_final[df_cliente_final[col_factura] == factura_final_id].iloc[0]

    st.info(f"Factura final seleccionada: **{factura_final[col_factura]}** "
            f"({factura_final['IMPORTE_CORRECTO']:,.2f} €)")

    # --- Selección de UTE (socios) ---
    opciones_utes = [c for c in opciones_clientes if c != cliente_final_cif]
    socio_cif = st.selectbox("Selecciona CIF de la UTE (socios)", opciones_utes)
    df_internas = df[df[col_cif] == socio_cif].copy()

    # --- Solver ---
    def cuadrar_internas(externa, df_internas):
        objetivo = int(externa['IMPORTE_CENT'])
        fecha_ref = externa[col_fecha_emision]

        data = list(zip(df_internas.index.tolist(),
                        df_internas['IMPORTE_CENT'].astype(int).tolist(),
                        (df_internas[col_fecha_emision] - fecha_ref).dt.days.fillna(0).astype(int).tolist()))

        n = len(data)
        if n == 0:
            return None

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"sel_{i}") for i in range(n)]

        # Suma exacta
        model.Add(sum(x[i] * data[i][1] for i in range(n)) == objetivo)

        # Minimizar número de facturas y diferencia de fechas
        costs = [abs(d[2]) for d in data]
        BIG_M = (max(costs) if costs else 0) * n + 1
        model.Minimize(BIG_M * sum(x) + sum(x[i] * costs[i] for i in range(n)))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            seleccion = [data[i][0] for i in range(n) if solver.Value(x[i]) == 1]
            return seleccion
        return None

    # --- Ejecutar búsqueda ---
    seleccion = cuadrar_internas(factura_final, df_internas)

    if seleccion:
        df_sel = df_internas.loc[seleccion].copy().sort_values(col_fecha_emision)
        st.success(f"✅ Se encontraron {len(df_sel)} facturas de la UTE que cuadran con la factura final {factura_final[col_factura]}")
        suma_sel = df_sel['IMPORTE_CORRECTO'].sum()
        st.write(f"**Suma seleccionada:** {suma_sel:,.2f} €")

        st.dataframe(df_sel)

        buffer = BytesIO()
        df_sel.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button(
            label="📥 Descargar facturas UTE asociadas",
            data=buffer,
            file_name=f"UTE_para_{factura_final[col_factura]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("❌ No se encontró una combinación EXACTA de facturas de la UTE para esta factura final")
