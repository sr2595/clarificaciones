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
    col_sociedad      = find_col(df, ['SOCIEDAD', 'Sociedad', 'SOC', 'EMPRESA'])
    col_cif_grupo     = find_col(df, ['CIF_GRUPO', 'Grupo', 'CIF Grupo'])

    faltan = []
    if not col_fecha_emision: faltan.append("fecha emisión")
    if not col_factura:       faltan.append("nº factura")
    if not col_importe:       faltan.append("importe")
    if not col_cif:           faltan.append("CIF")
    if not col_cif_grupo:     faltan.append("CIF grupo")
    if faltan:
        st.error("❌ No se pudieron localizar estas columnas: " + ", ".join(faltan))
        st.stop()

    # --- Normalizar ---
    df[col_fecha_emision] = pd.to_datetime(df[col_fecha_emision], dayfirst=True, errors='coerce')
    df[col_factura] = df[col_factura].astype(str)
    df[col_cif] = df[col_cif].astype(str)
    df['IMPORTE_CORRECTO'] = df[col_importe].apply(convertir_importe_europeo)
    df['IMPORTE_CENT'] = (df['IMPORTE_CORRECTO'] * 100).round().astype("Int64")

    # --- Opciones de clientes (CIF + Nombre), excluyendo UTES ---
    df_clientes_unicos = df[[col_cif, col_nombre_cliente]].drop_duplicates()
    df_clientes_unicos[col_cif] = df_clientes_unicos[col_cif].fillna("").str.strip()
    df_clientes_unicos[col_nombre_cliente] = df_clientes_unicos[col_nombre_cliente].fillna("").str.strip()
    # Excluir UTES: CIF con "U" después del prefijo L-00
    df_clientes_unicos = df_clientes_unicos[~df_clientes_unicos[col_cif].str[4:].str.upper().str.startswith("U")]
    df_clientes_unicos = df_clientes_unicos.sort_values(col_nombre_cliente)

    opciones_clientes = [
        f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
        for _, row in df_clientes_unicos.iterrows()
    ]
    mapping_cif = dict(zip(opciones_clientes, df_clientes_unicos[col_cif]))

    # --- Selección de cliente final ---
    cliente_final_display = st.selectbox("Selecciona cliente final (CIF - Nombre)", opciones_clientes)
    cliente_final_cif = mapping_cif[cliente_final_display]
    df_cliente_final = df[df[col_cif] == cliente_final_cif].copy()

    # --- Obtener grupo del cliente final ---
    cliente_final_grupo = df_cliente_final[col_cif_grupo].iloc[0]

    # --- Filtrar solo facturas de TSS ---
    df_tss = df_cliente_final[df_cliente_final[col_sociedad] == 'TSS']
    if df_tss.empty:
        st.warning("⚠️ No se encontraron facturas de TSS para este cliente final.")
        factura_final_display = None
        factura_final = None
    else:
        facturas_cliente = df_tss[[col_factura, col_fecha_emision, 'IMPORTE_CORRECTO']].dropna()
        opciones_facturas = [
            f"{row[col_factura]} - {row[col_fecha_emision].date()} - {row['IMPORTE_CORRECTO']:,.2f} €"
            for _, row in facturas_cliente.iterrows()
        ]
        factura_final_display = st.selectbox("Selecciona factura final TSS (90)", opciones_facturas)
        factura_final_id = factura_final_display.split(" - ")[0]
        factura_final = df_tss[df_tss[col_factura] == factura_final_id].iloc[0]

        st.info(f"Factura final seleccionada: **{factura_final[col_factura]}** "
                f"({factura_final['IMPORTE_CORRECTO']:,.2f} €)")

    # --- Selección de UTE (socios) solo UTES del mismo grupo ---
    df_utes_grupo = df[df[col_cif_grupo] == cliente_final_grupo].copy()
    df_utes_grupo = df_utes_grupo[df_utes_grupo[col_cif].str[4:].str.upper().str.startswith("U")]

    if df_utes_grupo.empty:
        st.warning("⚠️ No se encontraron UTES para este cliente final.")
        opciones_utes = []
        mapping_utes = {}
    else:
        df_utes_unicos = df_utes_grupo[[col_cif, col_nombre_cliente]].drop_duplicates()
        df_utes_unicos[col_nombre_cliente] = df_utes_unicos[col_nombre_cliente].fillna("").str.strip()
        opciones_utes = [
            f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
            for _, row in df_utes_unicos.iterrows()
        ]
        mapping_utes = dict(zip(opciones_utes, df_utes_unicos[col_cif]))

    socios_display = st.multiselect("Selecciona CIF(s) de la UTE (socios)", opciones_utes)
    socios_cifs = [mapping_utes[s] for s in socios_display]
    df_internas = df[df[col_cif].isin(socios_cifs)].copy()

    # --- Solver ---
    def cuadrar_internas(externa, df_internas):
        if externa is None or df_internas.empty:
            return None
        objetivo = int(externa['IMPORTE_CENT'])
        fecha_ref = externa[col_fecha_emision]

        seleccion_final = []
        for cif in df_internas[col_cif].unique():
            df_socio = df_internas[df_internas[col_cif] == cif].copy()
            data = list(zip(df_socio.index.tolist(),
                            df_socio['IMPORTE_CENT'].astype(int).tolist(),
                            (df_socio[col_fecha_emision] - fecha_ref).dt.days.fillna(0).astype(int).tolist()))
            n = len(data)
            if n == 0:
                continue

            model = cp_model.CpModel()
            x = [model.NewBoolVar(f"sel_{i}") for i in range(n)]
            model.Add(sum(x[i] * data[i][1] for i in range(n)) == objetivo)
            costs = [abs(d[2]) for d in data]
            BIG_M = (max(costs) if costs else 0) * n + 1
            model.Minimize(BIG_M * sum(x) + sum(x[i] * costs[i] for i in range(n)))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10
            status = solver.Solve(model)
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                seleccion_final.extend([data[i][0] for i in range(n) if solver.Value(x[i]) == 1])
        return seleccion_final if seleccion_final else None

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
