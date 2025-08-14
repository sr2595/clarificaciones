import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO
from datetime import datetime
import unicodedata, re

st.set_page_config(page_title="Clarificador PBI", page_icon="📄", layout="wide")
st.title("📄 Clarificador PBI")

# --------- Helpers robustos ---------
def _norm(texto):
    if texto is None:
        return ""
    s = str(texto)
    s = s.replace("\u00A0", " ")                # NBSP -> espacio normal
    s = unicodedata.normalize("NFKD", s)        # separa acentos
    s = "".join(c for c in s if not unicodedata.combining(c))  # elimina diacríticos
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)        # quita puntuación, deja espacios
    s = re.sub(r"\s+", " ", s).strip().lower()  # colapsa espacios, lower
    return s

def find_col(df, candidates):
    # Mapa de columna normalizada -> original
    norm_map = { _norm(c): c for c in df.columns }
    # 1) match exacto normalizado
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    # 2) match por "contiene" normalizado (más flexible)
    cand_norms = [_norm(c) for c in candidates]
    for orig in df.columns:
        n = _norm(orig)
        if any(cn in n or n in cn for cn in cand_norms if cn):
            return orig
    return None

# === Aquí la versión EXACTA que pediste ===
def convertir_importe_europeo(valor):
    if pd.isna(valor):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip().replace("€", "").replace(" ", "").replace('.', '').replace(',', '.')
    try:
        return float(texto)
    except:
        return None
# ===========================================

# --------- App ---------
archivo = st.file_uploader("Sube el archivo Excel", type=["xlsx", "xls"])
if archivo:
    try:
        df = pd.read_excel(archivo, engine="openpyxl")
    except Exception:
        df = pd.read_excel(archivo)

    # --- Mostrar columnas originales para depuración rápida ---
    with st.expander("🔎 Ver columnas detectadas en el Excel"):
        st.write(list(df.columns))

    # --- Detectar columnas de forma robusta ---
    col_fecha_emision = find_col(df, [
        'FECHA', 'Fecha', 'fecha', 'Fecha Emision', 'FECHA_EMISION', 'Fecha Emisión', 'FX_EMISION'
    ])
    col_factura = find_col(df, [
        'FACTURA', 'Factura', 'factura', 'Nº Factura', 'NRO_FACTURA', 'Núm.Doc.Deuda'
    ])
    col_importe = find_col(df, [
        'IMPORTE', 'Importe', 'importe', 'TOTAL', 'TOTAL_FACTURA'
    ])
    col_cif = find_col(df, [
        'T.Doc. - Núm.Doc.', 'T.Doc.-Núm.Doc.', 'T Doc - Num Doc', 
        'CIF', 'cif', 'NIF', 'nif', 'CIF_CLIENTE', 'NIF_CLIENTE', 'Cliente CIF', 'Cliente NIF'
    ])
    col_nombre_cliente = find_col(df, [
        'NOMBRE', 'Nombre', 'nombre',
        'CLIENTE', 'Cliente', 'cliente',
        'NOMBRE_CLIENTE', 'NOMBRE CLIENTE', 'Nombre Cliente', 'Nombre del Cliente',
        'Cliente Nombre', 'Cliente - Nombre', 'CLIENTE_NOMBRE',
        'RAZON_SOCIAL', 'Razón Social', 'Razon Social', 'RAZON SOCIAL'
    ])

    # --- Validación y feedback útil ---
    faltan = []
    if not col_fecha_emision: faltan.append("fecha emisión")
    if not col_factura:       faltan.append("nº factura")
    if not col_importe:       faltan.append("importe")
    if not col_cif:           faltan.append("T.Doc. - Núm.Doc. (CIF/NIF)")
    if faltan:
        st.error("❌ No se pudieron localizar estas columnas: " + ", ".join(faltan))
        st.info("Revisa el nombre exacto en el Excel o abre el desplegable de arriba para ver cómo llegan los encabezados.")
        st.stop()

    if not col_nombre_cliente:
        st.info("ℹ️ No se detectó la columna de *Nombre Cliente*. El selector mostrará solo el CIF/NIF.")

    # --- Procesar datos ---
    df[col_fecha_emision] = pd.to_datetime(df[col_fecha_emision], dayfirst=True, errors='coerce')
    df[col_factura] = df[col_factura].astype(str)
    df[col_cif] = df[col_cif].astype(str)
    if col_nombre_cliente:
        df[col_nombre_cliente] = df[col_nombre_cliente].astype(str)

    df['IMPORTE_CORRECTO'] = df[col_importe].apply(convertir_importe_europeo)

    total = df['IMPORTE_CORRECTO'].sum(skipna=True)
    minimo = df['IMPORTE_CORRECTO'].min(skipna=True)
    maximo = df['IMPORTE_CORRECTO'].max(skipna=True)

    st.write("**📊 Resumen del archivo:**")
    st.write(f"- Número total de facturas: {len(df)}")
    st.write(f"- Suma total importes: {total:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe mínimo: {minimo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe máximo: {maximo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

    # --- Desplegable: CIF (de T.Doc. - Núm.Doc.) + Nombre Cliente si existe ---
    if col_nombre_cliente:
        df_clientes_unicos = df[[col_cif, col_nombre_cliente]].drop_duplicates()
        df_clientes_unicos[col_nombre_cliente] = df_clientes_unicos[col_nombre_cliente].fillna("").str.strip()
        df_clientes_unicos[col_cif] = df_clientes_unicos[col_cif].fillna("").str.strip()
        # Ordenar por nombre para que sea más fácil identificar
        df_clientes_unicos = df_clientes_unicos.sort_values(col_nombre_cliente)
        opciones_clientes = [
            f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
            for _, row in df_clientes_unicos.iterrows()
        ]
        mapping_cif = dict(zip(opciones_clientes, df_clientes_unicos[col_cif]))
    else:
        opciones_clientes = sorted(df[col_cif].fillna("").str.strip().drop_duplicates())
        mapping_cif = {cif: cif for cif in opciones_clientes}

    cliente_seleccionado_display = st.selectbox("Selecciona cliente (CIF - Nombre)", opciones_clientes)
    cliente_cif = mapping_cif[cliente_seleccionado_display]

    # --- Inputs adicionales ---
    importe_objetivo = st.text_input("Introduce importe objetivo (ej: 295.206,63)")
    fecha_pago = st.date_input("Fecha de pago")

    # Filtrar por CIF cliente
    df_cliente = df[df[col_cif] == cliente_cif].copy()

    if importe_objetivo:
        # Validación e interpretación del importe
        try:
            importe_objetivo_eur = float(importe_objetivo.replace('.', '').replace(',', '.'))
            importe_objetivo_cent = int(round(importe_objetivo_eur * 100))
        except Exception:
            st.error("Formato de importe no válido.")
            st.stop()

        # Fechas base para desempate
        fecha_base = df_cliente[col_fecha_emision].min()
        if pd.isna(fecha_base):
            st.error("❌ La columna de fechas no contiene valores válidos para este cliente.")
            st.stop()

        df_cliente['DAYS_FROM_BASE'] = (df_cliente[col_fecha_emision] - fecha_base).dt.days.fillna(0).astype(int)
        df_cliente['IMPORTE_CENT'] = (df_cliente['IMPORTE_CORRECTO'] * 100).round().astype('Int64')

        # Filtrar facturas positivas y válidas
        df_positivas = df_cliente[(df_cliente['IMPORTE_CORRECTO'] > 0) & df_cliente['IMPORTE_CENT'].notna()].copy()
        if df_positivas.empty:
            st.warning("No hay facturas positivas con importes válidos para este cliente.")
            st.stop()

        # --- OR-Tools ---
        def seleccionar_facturas_exactas_ortools(df_filtrado, objetivo_cent, target_days):
            data = list(zip(
                df_filtrado.index.tolist(),
                df_filtrado['IMPORTE_CENT'].astype(int).tolist(),
                df_filtrado['DAYS_FROM_BASE'].astype(int).tolist()
            ))
            n = len(data)
            model = cp_model.CpModel()
            x = [model.NewBoolVar(f"sel_{i}") for i in range(n)]

            # Suma exacta
            model.Add(sum(x[i] * data[i][1] for i in range(n)) == int(objetivo_cent))

            # Objetivo: primero nº facturas, luego cercanía a la fecha objetivo
            if target_days is not None:
                costs = [abs(data[i][2] - target_days) for i in range(n)]
                max_cost = (max(costs) if costs else 0) * n
                BIG_M = max_cost + 1
                model.Minimize(BIG_M * sum(x) + sum(x[i] * costs[i] for i in range(n)))
            else:
                model.Minimize(sum(x))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10
            status = solver.Solve(model)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                return [data[i][0] for i in range(n) if solver.Value(x[i]) == 1]
            return None

        target_days = None
        if fecha_pago:
            try:
                target_days = (pd.to_datetime(datetime.combine(fecha_pago, datetime.min.time())) - fecha_base).days
            except Exception:
                target_days = None

        seleccion_idx = seleccionar_facturas_exactas_ortools(df_positivas, importe_objetivo_cent, target_days)

        if seleccion_idx:
            st.success(
                f"✅ Combinación encontrada para {importe_objetivo_eur:,.2f} €"
                .replace(",", "X").replace(".", ",").replace("X", ".")
            )
            df_sel = df_positivas.loc[seleccion_idx].copy()

            try:
                df_sel = df_sel.sort_values([col_fecha_emision, col_factura])
            except Exception:
                pass

            suma_sel = float(df_sel['IMPORTE_CORRECTO'].sum())
            st.write(
                f"**Suma seleccionada:** {suma_sel:,.2f} €"
                .replace(",", "X").replace(".", ",").replace("X", ".")
            )

            st.dataframe(df_sel)

            buffer = BytesIO()
            df_sel.to_excel(buffer, index=False, engine="openpyxl")
            buffer.seek(0)
            st.download_button(
                label="📥 Descargar facturas seleccionadas",
                data=buffer,
                file_name="facturas_seleccionadas_PBI.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("❌ No se encontró una combinación EXACTA de facturas.")
