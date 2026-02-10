import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO
from datetime import datetime
import unicodedata, re
import io
import os

st.write("DEBUG archivo en ejecución:", os.path.abspath(__file__))

st.set_page_config(page_title="Clarificador UTE con pagos", page_icon="📄", layout="wide")
st.title("📄 Clarificador UTE Masivo")

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

# --------- Inicializar variables globales ---------
factura_final = None
df_internas = pd.DataFrame()

# --------- Subida y normalización de PRISMA ---------
archivo_prisma = st.file_uploader("Sube el archivo PRISMA (CSV)", type=["csv"])
df_prisma = pd.DataFrame()

if archivo_prisma:
    try:
        # Leer PRISMA CSV
        df_prisma = pd.read_csv(
            archivo_prisma,
            sep=";",             # delimitador correcto
            skiprows=1,          # saltar las 2 primeras filas basura
            header=0,            # ahora la fila 3 es la cabecera real
            encoding="latin1",   # por si hay acentos
            on_bad_lines="skip"  # evita caídas si hay filas raras
        )
    except Exception as e:
        st.error(f"❌ Error leyendo PRISMA CSV: {e}")
        st.stop()

    # --- Detectar columnas ---
    col_id_ute_prisma       = find_col(df_prisma, ["id UTE"])
    col_num_factura_prisma  = find_col(df_prisma, ["Num. Factura", "Factura"])
    col_fecha_prisma        = find_col(df_prisma, ["Fecha Emisión", "Fecha"])
    col_cif_prisma          = find_col(df_prisma, ["CIF"])
    col_importe_prisma      = find_col(df_prisma, ["Total Base Imponible"])
    col_tipo_imp_prisma     = find_col(df_prisma, ["Tipo Impuesto"])
    col_cif_emisor_prisma   = find_col(df_prisma, ["CIF Emisor"])
    col_razon_social_prisma = find_col(df_prisma, ["Razón Social"])

    faltan = []
    for c, name in zip(
        [col_id_ute_prisma, col_num_factura_prisma, col_fecha_prisma, col_cif_prisma, col_importe_prisma],
        ["id UTE", "Num. Factura", "Fecha Emisión", "CIF", "Total Base Imponible"]
    ):
        if c is None:
            faltan.append(name)
    if faltan:
        st.error("❌ No se pudieron localizar estas columnas en PRISMA: " + ", ".join(faltan))
        st.stop()

    # --- Normalizar valores ---
    df_prisma[col_num_factura_prisma]  = df_prisma[col_num_factura_prisma].astype(str).str.strip()
    df_prisma[col_cif_prisma]          = df_prisma[col_cif_prisma].astype(str).str.replace(" ", "")
    df_prisma[col_id_ute_prisma]       = df_prisma[col_id_ute_prisma].astype(str).str.strip()
    df_prisma['IMPORTE_CORRECTO'] = df_prisma[col_importe_prisma].apply(convertir_importe_europeo)
    df_prisma['IMPORTE_CENT'] = (df_prisma['IMPORTE_CORRECTO'] * 100).round().astype("Int64")
    df_prisma[col_fecha_prisma] = pd.to_datetime(df_prisma[col_fecha_prisma], dayfirst=True, errors='coerce')
    df_prisma[col_tipo_imp_prisma] = df_prisma[col_tipo_imp_prisma].astype(str).str.strip().str.upper()
    col_tipo_imp_prisma = find_col(df_prisma, ["Tipo Impuesto"])
    if col_tipo_imp_prisma:
        # Limpiar espacios y convertir a mayúsculas para evitar errores de coincidencia
        df_prisma[col_tipo_imp_prisma] = df_prisma[col_tipo_imp_prisma].astype(str).str.strip().str.upper()
    else:
        st.error("❌ No se encontró la columna Tipo Impuesto en PRISMA")
        st.stop()

    st.success(f"✅ Archivo PRISMA cargado correctamente con {len(df_prisma)} filas")

    with st.expander("👀 Primeras filas PRISMA normalizado"):
        st.dataframe(df_prisma.head(10))

        # --- Debug: ver cómo quedan las facturas en PRISMA ---
    if not df_prisma.empty:
        st.subheader("🔍 Revisión columna de facturas en PRISMA")
        df_debug = df_prisma[[col_num_factura_prisma]].copy()
        # Añadimos una versión “normalizada” para comparar
        df_debug['FACTURA_NORMALIZADA'] = df_debug[col_num_factura_prisma].astype(str).str.strip().str.upper()
        st.dataframe(df_debug.head(20), use_container_width=True)
        
        # También ver si hay duplicados o espacios invisibles
        df_debug['LONGITUD'] = df_debug[col_num_factura_prisma].astype(str).str.len()
        df_debug['CONTIENE_ESPACIOS'] = df_debug[col_num_factura_prisma].astype(str).str.contains(" ")
        st.write("❗ Estadísticas rápidas:")
        st.write(f"- Número de filas: {len(df_debug)}")
        st.write(f"- Número de facturas únicas: {df_debug['FACTURA_NORMALIZADA'].nunique()}")
        st.write(f"- Facturas con espacios: {df_debug['CONTIENE_ESPACIOS'].sum()}")