import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO
from datetime import datetime
import unicodedata, re
import io
import os
import concurrent.futures
import time

st.write("DEBUG archivo en ejecución:", os.path.abspath(__file__))

st.set_page_config(page_title="Clarificador UTE con pagos", page_icon="📄", layout="wide")
st.title("📄 Clarificador UTE Masivo")

if "executor" not in st.session_state: 
    st.session_state.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) 
if "future" not in st.session_state: 
    st.session_state.future = None

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

def aplicar_impuestos_a_prisma(df_prisma, col_importe='IMPORTE_CORRECTO', col_tipo_impuesto='Tipo Impuesto'):
    factores = {
        "IGIC - 7": 1.07,
        "IPSIC - 10": 1.10,
        "IPSIM - 8": 1.08,
        "IVA - 0": 1.00,
        "IVA - 21": 1.21,
        "EXENTO": 1.0,
        "IVA - EXENTO": 1.0,
    }
    df_prisma[col_tipo_impuesto] = df_prisma[col_tipo_impuesto].astype(str).str.strip().str.upper()
    df_prisma['IMPORTE_CON_IMPUESTO'] = df_prisma.apply(
        lambda row: float(row[col_importe] * factores.get(row[col_tipo_impuesto], 1.0)),
        axis=1
    )
    return df_prisma

# --------- 1) Subida y normalización de PRISMA ---------
archivo_prisma = st.file_uploader("Sube el archivo PRISMA (CSV)", type=["csv"])

# Guardar bytes en session_state
if archivo_prisma is not None:
    st.session_state.prisma_bytes = archivo_prisma.getvalue()

# Si no hay bytes, no seguimos
if "prisma_bytes" not in st.session_state:
    st.stop()

# PROCESAR PRISMA SOLO UNA VEZ
if "df_prisma_procesado" not in st.session_state:
    st.info("⏳ Procesando archivo PRISMA por primera vez...")
    
    # Leer PRISMA desde bytes
    df_prisma = pd.read_csv(
        BytesIO(st.session_state.prisma_bytes),
        sep=";",
        skiprows=1,
        header=0,
        encoding="latin1",
        on_bad_lines="skip"
    )

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
    df_prisma['IMPORTE_CORRECTO']      = df_prisma[col_importe_prisma].apply(convertir_importe_europeo)
    df_prisma['IMPORTE_CENT']          = (df_prisma['IMPORTE_CORRECTO'] * 100).round().astype("Int64")
    df_prisma[col_fecha_prisma]        = pd.to_datetime(df_prisma[col_fecha_prisma], dayfirst=True, errors='coerce')

    # Normalizar tipo impuesto
    df_prisma[col_tipo_imp_prisma] = df_prisma[col_tipo_imp_prisma].astype(str).str.strip().str.upper()

    # Aplicar impuestos
    df_prisma = aplicar_impuestos_a_prisma(df_prisma, col_tipo_impuesto=col_tipo_imp_prisma)
    
    # 🧹 Liberar bytes crudos
    del st.session_state.prisma_bytes  

    # Guardar en session_state
    st.session_state.df_prisma_procesado = df_prisma
    st.session_state.col_num_factura_prisma = col_num_factura_prisma
    st.session_state.col_cif_prisma = col_cif_prisma
    st.session_state.col_id_ute_prisma = col_id_ute_prisma
    st.session_state.col_tipo_imp_prisma = col_tipo_imp_prisma
    
    st.success(f"✅ Archivo PRISMA cargado correctamente con {len(df_prisma)} filas")
else:
    # Recuperar desde session_state
    df_prisma = st.session_state.df_prisma_procesado
    col_num_factura_prisma = st.session_state.col_num_factura_prisma
    col_cif_prisma = st.session_state.col_cif_prisma
    col_id_ute_prisma = st.session_state.col_id_ute_prisma
    col_tipo_imp_prisma = st.session_state.col_tipo_imp_prisma
    
    st.success(f"✅ Archivo PRISMA ya cargado ({len(df_prisma)} filas)")

with st.expander("👀 Primeras filas PRISMA normalizado"):
    st.dataframe(df_prisma.head(10))

# --- Debug: ver cómo quedan las facturas en PRISMA ---
if not df_prisma.empty:
    with st.expander("🔍 Revisión columna de facturas en PRISMA"):
        df_debug = df_prisma[[col_num_factura_prisma]].copy()
        df_debug['FACTURA_NORMALIZADA'] = df_debug[col_num_factura_prisma].astype(str).str.strip().str.upper()
        st.dataframe(df_debug.head(20), use_container_width=True)
        
        df_debug['LONGITUD'] = df_debug[col_num_factura_prisma].astype(str).str.len()
        df_debug['CONTIENE_ESPACIOS'] = df_debug[col_num_factura_prisma].astype(str).str.contains(" ")
        st.write("❗ Estadísticas rápidas:")
        st.write(f"- Número de filas: {len(df_debug)}")
        st.write(f"- Número de facturas únicas: {df_debug['FACTURA_NORMALIZADA'].nunique()}")
        st.write(f"- Facturas con espacios: {df_debug['CONTIENE_ESPACIOS'].sum()}")

    with st.expander("🔍 Debug: revisión de importes con impuesto aplicado"):
        st.dataframe(
            df_prisma[[col_num_factura_prisma, col_cif_prisma, 'IMPORTE_CORRECTO', col_tipo_imp_prisma, 'IMPORTE_CON_IMPUESTO']].head(20),
            use_container_width=True
        )
        st.write(f"- Total importe original: {df_prisma['IMPORTE_CORRECTO'].sum():,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
        st.write(f"- Total importe con impuesto: {df_prisma['IMPORTE_CON_IMPUESTO'].sum():,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

# --------- 2) Subida del Maestro UTEs ---------
archivo_maestro = st.file_uploader("Sube el Maestro UTEs (.xlsx — guarda desde Excel como xlsx)", type=["xlsx"], key="maestro")

if archivo_maestro is not None:
    st.session_state.maestro_bytes = archivo_maestro.getvalue()
    st.session_state.maestro_nombre = archivo_maestro.name
    if 'df_maestro_utes' in st.session_state:
        del st.session_state['df_maestro_utes']

if "df_maestro_utes" not in st.session_state and "maestro_bytes" in st.session_state:
    st.info("⏳ Procesando Maestro UTEs...")
    nombre_m = st.session_state.get('maestro_nombre', '')
    m_bytes = st.session_state.maestro_bytes

    # Maestro UTEs — leer xlsx directamente
    try:
        df_maestro = pd.read_excel(BytesIO(m_bytes), sheet_name="Datos", engine="openpyxl")
    except Exception as e:
        st.error(f"❌ Error leyendo Maestro UTEs: {e}")
        st.stop()

    col_ute_m   = find_col(df_maestro, ['UTE', 'CIF UTE', 'CIF_UTE'])
    col_tde_m   = find_col(df_maestro, ['Porc. TdE', 'Porc TdE', 'TDE', 'PORC_TDE'])
    col_tme_m   = find_col(df_maestro, ['Porc. TME', 'Porc TME', 'TME', 'PORC_TME'])
    col_tsol_m  = find_col(df_maestro, ['Porc. TSOL', 'Porc TSOL', 'TSOL', 'PORC_TSOL'])
    col_otros_m = find_col(df_maestro, ['Porc. Otros', 'Porc Otros', 'Otros', 'PORC_OTROS'])

    faltan_m = [n for c, n in [(col_ute_m,'UTE'),(col_tde_m,'Porc TdE'),(col_tme_m,'Porc TME')] if c is None]
    if faltan_m:
        st.error(f"❌ Maestro UTEs: no se encontraron columnas: {', '.join(faltan_m)}")
    else:
        df_maestro[col_ute_m] = df_maestro[col_ute_m].astype(str).str.strip().str.upper()
        for col in [c for c in [col_tde_m, col_tme_m, col_tsol_m, col_otros_m] if c]:
            df_maestro[col] = pd.to_numeric(
                df_maestro[col].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            ).fillna(0.0)

        maestro_map = {}
        for _, row in df_maestro.iterrows():
            cif = str(row[col_ute_m]).strip().upper()
            maestro_map[cif] = {
                'TDE':   float(row[col_tde_m])   if col_tde_m  else 0.0,
                'TME':   float(row[col_tme_m])   if col_tme_m  else 0.0,
                'TSOL':  float(row[col_tsol_m])  if col_tsol_m else 0.0,
                'OTROS': float(row[col_otros_m]) if col_otros_m else 0.0,
            }

        st.session_state.df_maestro_utes = df_maestro
        st.session_state.maestro_map = maestro_map
        del st.session_state.maestro_bytes
        st.success(f"✅ Maestro UTEs cargado: {len(maestro_map)} UTEs")
        with st.expander("👀 Primeras filas Maestro UTEs"):
            st.dataframe(df_maestro.head(10))

elif "df_maestro_utes" in st.session_state:
    st.success(f"✅ Maestro UTEs ya cargado ({len(st.session_state.maestro_map)} UTEs)")

# --------- 3) Subida de archivo de pagos (Cruce_Movs) ---------
cobros_file = st.file_uploader(
    "Sube el Excel de pagos de UTE ej. Informe_Cruce_Movimientos 19052025 a 19082025",
    type=['xlsm', 'xlsx', 'csv'],
    key="cobros"
)

# Guardar bytes en session_state
if cobros_file is not None:
    st.session_state.cobros_bytes = cobros_file.getvalue()

# Si no hay bytes, no seguimos
if "cobros_bytes" not in st.session_state and "df_cobros_procesado" not in st.session_state:
    st.stop()

# PROCESAR PAGOS SOLO UNA VEZ
if "df_cobros_procesado" not in st.session_state:
    st.info("⏳ Procesando archivo de PAGOS por primera vez...")
    
    # Leer PAGOS desde bytes
    data = BytesIO(st.session_state.cobros_bytes)

    # Detectar hoja
    xls = pd.ExcelFile(data, engine="openpyxl")
    sheet = "Cruce_Movs" if "Cruce_Movs" in xls.sheet_names else xls.sheet_names[0]

    # Reset pointer y leer
    data.seek(0)
    df_cobros = pd.read_excel(data, sheet_name=sheet, engine="openpyxl")

    # Normalizar columnas
    df_cobros.columns = (
        df_cobros.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'[áàäâ]', 'a', regex=True)
        .str.replace(r'[éèëê]', 'e', regex=True)
        .str.replace(r'[íìïî]', 'i', regex=True)
        .str.replace(r'[óòöô]', 'o', regex=True)
        .str.replace(r'[úùüû]', 'u', regex=True)
        .str.replace(r'[^0-9a-z]', '_', regex=True)
        .str.replace(r'__+', '_', regex=True)
        .str.strip('_')
    )

    # Mapeo seguro
    col_map = {
        'fec_operacion': ['fec_operacion', 'fecha_operacion', 'fec_oper'],
        'importe': ['importe', 'imp', 'monto', 'amount', 'valor'],
        'posible_factura': ['posible_factura', 'factura', 'posiblefactura'],
        'norma_43': ['norma_43', 'norma43'],
        'CIF_UTE': ['cif', 'cif_ute'],
        'denominacion': ['denominacion', 'nombre', 'razon_social', 'nombre_ute']
    }

    for target, possibles in col_map.items():
        for p in possibles:
            if p in df_cobros.columns:
                df_cobros.rename(columns={p: target}, inplace=True)
                break

    # Tipos correctos
    if 'fec_operacion' in df_cobros.columns:
        df_cobros['fec_operacion'] = pd.to_datetime(df_cobros['fec_operacion'], errors='coerce')

    if 'importe' in df_cobros.columns:
        df_cobros['importe'] = pd.to_numeric(df_cobros['importe'], errors='coerce')

    if 'posible_factura' in df_cobros.columns:
        df_cobros['posible_factura'] = df_cobros['posible_factura'].astype(str).str.strip()

    if 'norma_43' in df_cobros.columns:
        df_cobros['norma_43'] = df_cobros['norma_43'].astype(str).str.strip()

    if 'CIF_UTE' in df_cobros.columns:
        df_cobros['CIF_UTE'] = df_cobros['CIF_UTE'].astype(str).str.strip()

    # Guardar en session_state
    st.session_state.df_cobros_procesado = df_cobros
    
    # Estadísticas básicas
    num_filas = len(df_cobros)
    total_importes = df_cobros['importe'].sum(skipna=True)
    min_importe = df_cobros['importe'].min(skipna=True)
    max_importe = df_cobros['importe'].max(skipna=True)
    pagos_con_factura = df_cobros['posible_factura'].notna().sum() if 'posible_factura' in df_cobros.columns else 0

    st.write("**📊 Resumen archivo PAGOS:**")
    st.write(f"- Número de filas: {num_filas}")
    st.write(f"- Suma total de importes: {total_importes:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe mínimo: {min_importe:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe máximo: {max_importe:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Pagos con posible factura: {pagos_con_factura}")

    st.dataframe(df_cobros.head(10), use_container_width=True)
else:
    # Recuperar desde session_state
    df_cobros = st.session_state.df_cobros_procesado
    st.success(f"✅ Archivo PAGOS ya cargado ({len(df_cobros)} filas)")

#####--- 4) PEDIR FECHAS PARA FILTRAR PAGOS ---#####

if not df_cobros.empty:
    st.subheader("🔹 Selecciona el día para el cruce de pagos")
    
    st.info("💡 **Tip**: Puedes cambiar de día cuantas veces quieras. El cruce solo se ejecutará cuando pulses el botón.")

    # Pedir solo un día
    df_cobros['fec_operacion'] = df_cobros['fec_operacion'].dt.normalize()
    dias_disponibles = sorted(df_cobros['fec_operacion'].dropna().unique())
    fecha_seleccionada = st.selectbox("Selecciona el día:", dias_disponibles)

    # Filtrar pagos del día seleccionado
    df_cobros_filtrado = df_cobros[
        df_cobros['fec_operacion'].notna() &
        (df_cobros['fec_operacion'].dt.normalize() == pd.to_datetime(fecha_seleccionada))
    ].copy()

    st.write(f"ℹ️ Pagos del día seleccionado: {len(df_cobros_filtrado)}")

    # Extraer solo las columnas necesarias para el cruce
    columnas_cruce = ['fec_operacion', 'importe', 'posible_factura', 'CIF_UTE']
    if 'norma_43' in df_cobros_filtrado.columns:
        columnas_cruce.append('norma_43')
    if 'denominacion' in df_cobros_filtrado.columns:
        columnas_cruce.append('denominacion')

    df_pagos = df_cobros_filtrado[columnas_cruce].copy()

    st.subheader("🔍 Pagos filtrados para cruce")
    st.dataframe(df_pagos.head(10), use_container_width=True)
    st.write(f"Total importes en el día: {df_pagos['importe'].sum():,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

    #######--- 5) PREPARAR DATOS PARA CRUCE ---#######

    # PASO A: Preparar df_prisma_90 base SOLO UNA VEZ (sin filtro COBRA)
    if "df_prisma_90_base" not in st.session_state:
        st.info("⏳ Preparando datos de PRISMA para cruce (solo la primera vez)...")
        
        df_prisma['CIF'] = (
            df_prisma['CIF']
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
            .str.upper()
        )
        df_prisma[col_num_factura_prisma] = (
            df_prisma[col_num_factura_prisma]
            .astype(str)
            .str.strip()
        )

        df_temp = df_prisma.copy()
        df_temp[col_num_factura_prisma] = df_temp[col_num_factura_prisma].astype(str).str.strip()
        df_temp['Id UTE'] = df_temp['Id UTE'].astype(str).str.strip()
        df_temp['CIF'] = df_temp['CIF'].astype(str).str.strip()

        df_sin_90 = df_temp[~df_temp[col_num_factura_prisma].str.startswith("90")].copy()
        cif_por_ute = df_sin_90.groupby('Id UTE')['CIF'].first().to_dict()

        df_prisma_90_base = df_temp[df_temp[col_num_factura_prisma].str.startswith("90")].copy()
        df_prisma_90_base['Id UTE'] = df_prisma_90_base['Id UTE'].astype(str).str.strip()
        df_prisma_90_base['CIF_UTE_REAL'] = df_prisma_90_base['Id UTE'].apply(lambda x: cif_por_ute.get(x, "NONE"))
        
        # Columna CIF original normalizado (para cruce con COBRA)
        df_prisma_90_base['CIF_Original_Norm'] = df_prisma_90_base['CIF'].astype(str).str.replace(" ", "").str.strip().str.upper()
        df_prisma_90_base['Num_Factura_Norm'] = df_prisma_90_base[col_num_factura_prisma].astype(str).str.strip().str.upper()

        if 'Nombre UTE' in df_prisma.columns:
            df_prisma_90_base['Nombre_UTE'] = df_prisma_90_base['Id UTE'].map(df_prisma.set_index('Id UTE')['Nombre UTE'])
        else:
            df_prisma_90_base['Nombre_UTE'] = "DESCONOCIDO"

        if 'Nombre Cliente' in df_prisma.columns:
            df_prisma_90_base['Nombre_Cliente'] = df_prisma_90_base['CIF_UTE_REAL'].map(df_prisma.set_index('CIF')['Nombre Cliente'])
        else:
            df_prisma_90_base['Nombre_Cliente'] = "DESCONOCIDO"

        st.session_state.df_prisma_90_base = df_prisma_90_base
        st.success(f"✅ Base PRISMA preparada: {len(df_prisma_90_base)} facturas 90")
    else:
        df_prisma_90_base = st.session_state.df_prisma_90_base
        st.success(f"✅ Base PRISMA ya cargada ({len(df_prisma_90_base)} facturas 90)")

    # PASO B simplificado: df_prisma_90 = df_prisma_90_base directamente
    # Sin COBRA — usamos las 90s de PRISMA como base directa
    if "df_prisma_90_preparado" not in st.session_state:
        df_prisma_90 = df_prisma_90_base.copy()
        df_prisma_90['CIF_ORIGINAL'] = df_prisma_90['CIF'].astype(str).str.strip()
        if 'Fecha Emisión' not in df_prisma_90.columns:
            col_f = find_col(df_prisma_90, ['Fecha Emisión', 'Fecha Emision', 'Fecha'])
            if col_f:
                df_prisma_90['Fecha Emisión'] = df_prisma_90[col_f]
            else:
                df_prisma_90['Fecha Emisión'] = pd.NaT

        # Filtrar solo con importe positivo
        df_prisma_90 = df_prisma_90[df_prisma_90['IMPORTE_CON_IMPUESTO'] > 0].copy()

        with st.expander(f"🔍 Facturas 90 de PRISMA ({len(df_prisma_90)})"):
            st.dataframe(df_prisma_90[['Num_Factura_Norm','CIF_UTE_REAL','Id UTE','IMPORTE_CON_IMPUESTO','Fecha Emisión']].head(20))

        # Facturas negativas informativo
        facturas_negativas = df_prisma_90_base[df_prisma_90_base['IMPORTE_CON_IMPUESTO'] < 0]
        if len(facturas_negativas) > 0:
            with st.expander(f"⚠️ Facturas 90 NEGATIVAS ignoradas ({len(facturas_negativas)})"):
                st.dataframe(facturas_negativas[['Num_Factura_Norm','CIF_UTE_REAL','IMPORTE_CON_IMPUESTO']].sort_values('IMPORTE_CON_IMPUESTO'))

        st.session_state.df_prisma_90_preparado = df_prisma_90
        st.success(f"✅ Facturas 90 de PRISMA listas: {len(df_prisma_90)}")
    else:
        df_prisma_90 = st.session_state.df_prisma_90_preparado
        st.success(f"✅ Facturas 90 ya cargadas ({len(df_prisma_90)} facturas)")

        # -------------------------------
    # 3️⃣ FUNCIÓN OR-TOOLS CON SOCIOS (solo PRISMA + Maestro UTEs)
    # -------------------------------
    def cruzar_pagos_con_prisma_exacto(
        df_pagos, df_prisma_90, df_prisma_completo,
        col_num_factura_prisma,
        tolerancia=0.01, maestro_map=None
    ):
        """
        Flujo simplificado — sin COBRA para socios:
        1. Para cada pago, buscar 90s en PRISMA que sumen el importe
           (priorizando posible_factura si está informada)
        2. Para cada 90 seleccionada → buscar socios TDE/TME en PRISMA
        3. La diferencia entre importe_90 y suma_socios_PRISMA se desglosa
           como TSOL u OTROS según los porcentajes del Maestro UTEs
        """
        if maestro_map is None:
            maestro_map = {}
        resultados = []

        # ── Fecha de columna de socios en PRISMA ────────────────────────────────
        col_fecha_factura = None
        for posible_col in ['Fecha Emisión', 'Fecha Emision', 'FECHA_EMISION', 'Fecha']:
            if posible_col in df_prisma_completo.columns:
                col_fecha_factura = posible_col
                break

        # ── Socios PRISMA por Id UTE (facturas no-90 con importe positivo) ───────
        df_socios_raw = df_prisma_completo[
            (~df_prisma_completo[col_num_factura_prisma].astype(str).str.startswith("90")) &
            (df_prisma_completo['IMPORTE_CON_IMPUESTO'] > 0)
        ].copy()

        # Identificar sociedad por prefijo de número de factura:
        #   60... → TDE (fijo)
        #   ADM... → TME (móvil)
        #   otros → OTROS
        def detectar_sociedad_prisma(num_factura):
            n = str(num_factura).strip().upper()
            if n.startswith('60'):
                return 'TDE'
            elif n.startswith('ADM'):
                return 'TME'
            else:
                return 'OTROS'

        df_socios_raw['SOCIEDAD_PRISMA'] = df_socios_raw[col_num_factura_prisma].apply(detectar_sociedad_prisma)

        socios_por_ute = {}
        for id_ute, grupo in df_socios_raw.groupby('Id UTE'):
            cols_s = [col_num_factura_prisma, 'IMPORTE_CON_IMPUESTO', 'CIF', 'SOCIEDAD_PRISMA']
            if col_fecha_factura:
                cols_s.append(col_fecha_factura)
            socios_por_ute[str(id_ute).strip()] = grupo[cols_s].copy()
        del df_socios_raw

        # ── Agrupar facturas 90 por CIF_UTE_REAL ─────────────────────────────────
        # Indexar con U y con J para cubrir la discrepancia entre PRISMA (U) y pagos (J)
        facturas_por_cif = {}
        for cif, g in df_prisma_90.groupby('CIF_UTE_REAL'):
            facturas_por_cif[cif] = g.copy()
            cif_str = str(cif)
            if cif_str.startswith('U'):
                facturas_por_cif['J' + cif_str[1:]] = g.copy()
            elif cif_str.startswith('J'):
                facturas_por_cif['U' + cif_str[1:]] = g.copy()

        # ── Lookup num_factura → fila (para posible_factura) ─────────────────────
        todas_90_por_num = {
            str(row['Num_Factura_Norm']).strip().upper(): row
            for _, row in df_prisma_90.iterrows()
        }

        # ── Bucle de pagos ────────────────────────────────────────────────────────
        for idx, pago in df_pagos.iterrows():
            try:
                cif_pago     = pago['CIF_UTE']
                importe_pago = pago['importe']
                fecha_pago   = pago['fec_operacion']

                # Obtener porcentajes del maestro para esta UTE
                cif_j = str(cif_pago).strip().upper()
                cif_u = ('U' + cif_j[1:]) if cif_j.startswith('J') else cif_j
                porcentajes = maestro_map.get(cif_j) or maestro_map.get(cif_u) or {}

                # ── PRE-PASO: posible_factura ────────────────────────────────────
                posible_num = str(pago.get('posible_factura', '')).strip().upper()
                forzar_90 = None
                if posible_num and posible_num not in ('', 'NAN', 'NONE', 'N/A'):
                    if posible_num in todas_90_por_num:
                        forzar_90 = todas_90_por_num[posible_num]

                # ── Verificar que hay 90s para este CIF ─────────────────────────
                if cif_pago not in facturas_por_cif and forzar_90 is None:
                    resultados.append({
                        'CIF_UTE': cif_pago, 'fecha_pago': fecha_pago, 'importe_pago': importe_pago,
                        'facturas_90_asignadas': 'SIN_90s_PARA_ESTE_CIF',
                        'importe_facturas_90': 0.0, 'desglose_facturas_90': None,
                        'diferencia_pago_vs_90': importe_pago,
                        'advertencia': f'No existen facturas 90 en PRISMA para CIF {cif_pago}'
                    })
                    continue

                # ── Construir df_facturas ────────────────────────────────────────
                TOLERANCIA_90 = max(tolerancia, 2.0)

                if forzar_90 is not None:
                    # posible_factura encontrada → usar directamente
                    forzar_row = forzar_90 if isinstance(forzar_90, dict) else forzar_90.to_dict()
                    df_facturas = pd.DataFrame([forzar_row])
                    for col_req, val_def in [('IMPORTE_CON_IMPUESTO', 0.0), ('Fecha Emisión', pd.NaT),
                                              ('TIENE_MATCH_PRISMA', True),
                                              ('CIF_ORIGINAL', ''), ('Num_Factura_Norm', posible_num)]:
                        if col_req not in df_facturas.columns:
                            df_facturas[col_req] = val_def
                    # Id UTE: intentar obtenerlo de df_prisma_90 si no está
                    if 'Id UTE' not in df_facturas.columns or str(df_facturas['Id UTE'].iloc[0]).strip() in ('', 'DESCONOCIDO', 'nan'):
                        if posible_num in todas_90_por_num:
                            id_ute_real = todas_90_por_num[posible_num].get('Id UTE', 'DESCONOCIDO')
                            df_facturas['Id UTE'] = str(id_ute_real).strip()
                else:
                    df_todas = facturas_por_cif[cif_pago].copy()
                    # Filtrar por importe y fecha
                    df_cands = df_todas[
                        (df_todas['IMPORTE_CON_IMPUESTO'] > 0) &
                        (df_todas['IMPORTE_CON_IMPUESTO'] <= importe_pago + TOLERANCIA_90) &
                        (df_todas['Fecha Emisión'].isna() | (df_todas['Fecha Emisión'] <= fecha_pago))
                    ].copy()

                    # Filtrar por coherencia con maestro: la 90 debe tener socios
                    # que cubran todas las sociedades con porcentaje > 0 en el maestro
                    socs_requeridas = {soc for soc, porc in porcentajes.items() if porc and porc > 0 and soc not in ('TSOL', 'OTROS')}
                    if socs_requeridas and not df_cands.empty:
                        candidatas_validas = []
                        for _, f90 in df_cands.iterrows():
                            id_ute_90 = str(f90.get('Id UTE', 'DESCONOCIDO')).strip()
                            if id_ute_90 in socios_por_ute:
                                df_soc_90 = socios_por_ute[id_ute_90]
                                if col_fecha_factura and col_fecha_factura in df_soc_90.columns and pd.notna(f90.get('Fecha Emisión')):
                                    df_soc_90 = df_soc_90[df_soc_90[col_fecha_factura] <= f90['Fecha Emisión']]
                                socs_disponibles = set(df_soc_90['SOCIEDAD_PRISMA'].unique())
                                # Solo rechazar si hay socios pero no cubren los requeridos
                                if socs_disponibles and not socs_requeridas.issubset(socs_disponibles):
                                    continue  # esta 90 no tiene los socios requeridos
                            candidatas_validas.append(f90)
                        df_facturas = pd.DataFrame(candidatas_validas) if candidatas_validas else pd.DataFrame()
                    else:
                        df_facturas = df_cands

                    if df_facturas.empty:
                        # Explicar por qué se descartaron
                        razones = []
                        for _, f90 in df_todas.iterrows():
                            imp = f90['IMPORTE_CON_IMPUESTO']
                            fec = f90.get('Fecha Emisión', pd.NaT)
                            num = f90['Num_Factura_Norm']
                            if imp > 0 and imp <= importe_pago + TOLERANCIA_90:
                                fec_str = fec.strftime('%d/%m/%Y') if pd.notna(fec) else 'sin fecha'
                                razones.append(f"{num} ({imp:.2f}€): FECHA POSTERIOR AL PAGO ({fec_str} > {fecha_pago.strftime('%d/%m/%Y')})")
                            else:
                                razones.append(f"{num} ({imp:.2f}€): importe no encaja con pago {importe_pago:.2f}€ (dif={round(imp-importe_pago,2):+.2f}€)")
                        resultados.append({
                            'CIF_UTE': cif_pago, 'fecha_pago': fecha_pago, 'importe_pago': importe_pago,
                            'facturas_90_asignadas': 'SIN_COMBINACION_VALIDA',
                            'importe_facturas_90': 0.0, 'desglose_facturas_90': None,
                            'diferencia_pago_vs_90': importe_pago,
                            'advertencia': 'Ningun 90 candidata: ' + ' || '.join(razones) if razones else f'Sin 90s para pago {importe_pago:.2f}€'
                        })
                        continue

                # ── Solver OR-Tools: qué 90s suman el importe del pago ───────────
                df_facturas = df_facturas.sort_values(['Fecha Emisión', 'IMPORTE_CON_IMPUESTO'], ascending=[True, True])
                numeros_facturas  = df_facturas['Num_Factura_Norm'].tolist()
                importes_facturas = df_facturas['IMPORTE_CON_IMPUESTO'].tolist()
                ids_ute           = df_facturas['Id UTE'].tolist()
                fechas_90         = df_facturas['Fecha Emisión'].tolist()

                n             = len(importes_facturas)
                pagos_cent    = int(round(importe_pago * 100))
                facturas_cent = [int(round(f * 100)) for f in importes_facturas]
                tol_cent      = int(round(tolerancia * 100))

                model  = cp_model.CpModel()
                x      = [model.NewBoolVar(f"x_{i}") for i in range(n)]
                model.Add(sum(x[i] * facturas_cent[i] for i in range(n)) >= pagos_cent - tol_cent)
                model.Add(sum(x[i] * facturas_cent[i] for i in range(n)) <= pagos_cent + tol_cent)
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 3
                solver.parameters.log_search_progress = False
                status = solver.Solve(model)

                if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                    suma = sum(importes_facturas)
                    detalle = ' | '.join([f"{numeros_facturas[i]} ({importes_facturas[i]:.2f}€)" for i in range(n)])
                    resultados.append({
                        'CIF_UTE': cif_pago, 'fecha_pago': fecha_pago, 'importe_pago': importe_pago,
                        'facturas_90_asignadas': 'SIN_COMBINACION_EXACTA',
                        'importe_facturas_90': 0.0, 'desglose_facturas_90': None,
                        'diferencia_pago_vs_90': importe_pago,
                        'advertencia': f'No cuadra: suma disponible={suma:.2f}€ vs pago={importe_pago:.2f}€. 90s: {detalle}'
                    })
                    continue

                # ── Para cada 90 seleccionada: buscar socios en PRISMA ───────────
                seleccion = [i for i in range(n) if solver.Value(x[i]) == 1]
                desglose_por_factura_90 = []
                importe_facturas_90     = 0.0

                for i in seleccion:
                    num_90      = numeros_facturas[i]
                    imp_90      = importes_facturas[i]
                    id_ute      = str(ids_ute[i]).strip()
                    fecha_90    = fechas_90[i]
                    importe_facturas_90 += imp_90

                    # Buscar socios TDE/TME en PRISMA para este Id UTE
                    socios_prisma       = []
                    importe_socios_prisma = 0.0

                    if id_ute in socios_por_ute:
                        df_soc = socios_por_ute[id_ute].copy()
                        for _, socio in df_soc.iterrows():
                            sociedad = str(socio.get('SOCIEDAD_PRISMA', 'OTROS'))
                            socios_prisma.append({
                                'num_factura': str(socio[col_num_factura_prisma]),
                                'cif':         str(socio['CIF']),
                                'importe':     float(socio['IMPORTE_CON_IMPUESTO']),
                                'fuente':      f'PRISMA ({sociedad})'
                            })
                            importe_socios_prisma += float(socio['IMPORTE_CON_IMPUESTO'])

                    # ── Calcular diferencia y desglosarla con el Maestro UTEs ────
                    diferencia = round(imp_90 - importe_socios_prisma, 2)
                    socios_estimados = []  # socios calculados por % del maestro que no están en PRISMA

                    if abs(diferencia) > tolerancia and porcentajes:
                        # Los socios TDE/TME ya están en PRISMA.
                        # La diferencia restante corresponde a TSOL u OTROS según el maestro.
                        socs_en_prisma = {s['fuente'].split('(')[-1].rstrip(')') for s in socios_prisma}
                        for soc_nombre, porc in porcentajes.items():
                            if soc_nombre in socs_en_prisma:
                                continue  # ya está cubierto por PRISMA
                            if porc and porc > 0:
                                imp_estimado = round(imp_90 * porc / 100.0, 2)
                                socios_estimados.append({
                                    'num_factura': 'PENDIENTE',
                                    'cif':         soc_nombre,
                                    'importe':     imp_estimado,
                                    'fuente':      f'ESTIMADO_MAESTRO ({soc_nombre} {porc:.1f}%)'
                                })

                    diferencia_final = round(imp_90 - importe_socios_prisma - sum(s['importe'] for s in socios_estimados), 2)

                    # Estado: si hay diferencia sin explicar → advertir
                    if abs(diferencia_final) > tolerancia:
                        estado = f"⚠️ Diferencia sin cubrir: {diferencia_final:.2f}€"
                    elif socios_estimados:
                        estado = f"✅ Socios PRISMA + estimados por Maestro ({', '.join(s['cif'] for s in socios_estimados)})"
                    else:
                        estado = "✅ Cuadra con socios PRISMA"

                    desglose_por_factura_90.append({
                        'factura_90':            num_90,
                        'importe_90':            imp_90,
                        'caso':                  'PRISMA',
                        'cif_cliente_final':     df_facturas.iloc[i].get('CIF_ORIGINAL', '') if i < len(df_facturas) else '',
                        'socios':                socios_prisma + socios_estimados,
                        'importe_socios':        importe_socios_prisma + sum(s['importe'] for s in socios_estimados),
                        'diferencia_90_socios':  diferencia_final,
                        'socios_prisma':         socios_prisma,
                        'socios_cobra':          socios_estimados,
                        'importe_socios_prisma': importe_socios_prisma,
                        'importe_socios_cobra':  sum(s['importe'] for s in socios_estimados),
                        'estado_cobra':          estado
                    })

                facturas_90_str       = ', '.join([d['factura_90'] for d in desglose_por_factura_90])
                diferencia_pago_vs_90 = round(importe_pago - importe_facturas_90, 2)

                # Advertencia si hay diferencias sin cubrir
                difs_sin_cubrir = [d for d in desglose_por_factura_90 if abs(d['diferencia_90_socios']) > tolerancia]
                advertencia = None
                if difs_sin_cubrir:
                    advertencia = ' | '.join([f"{d['factura_90']}: dif={d['diferencia_90_socios']:.2f}€" for d in difs_sin_cubrir])

                resultados.append({
                    'CIF_UTE': cif_pago, 'fecha_pago': fecha_pago, 'importe_pago': importe_pago,
                    'facturas_90_asignadas': facturas_90_str, 'importe_facturas_90': importe_facturas_90,
                    'desglose_facturas_90': desglose_por_factura_90,
                    'diferencia_pago_vs_90': diferencia_pago_vs_90, 'advertencia': advertencia
                })

            except Exception as e:
                resultados.append({
                    'CIF_UTE': pago.get('CIF_UTE', 'ERROR'), 'fecha_pago': pago.get('fec_operacion'),
                    'importe_pago': pago.get('importe', 0),
                    'facturas_90_asignadas': f"ERROR: {str(e)}", 'importe_facturas_90': 0.0,
                    'desglose_facturas_90': None, 'diferencia_pago_vs_90': pago.get('importe', 0), 'advertencia': None
                })
                continue

        return pd.DataFrame(resultados)

    # -------------------------------
    # 4️⃣ BOTÓN PARA EJECUTAR EL SOLVER
    # -------------------------------

    st.markdown("---")
    st.subheader("🚀 Ejecutar cruce de pagos con facturas")

    col_tol, _ = st.columns([1, 2])
    with col_tol:
        tolerancia_centimos = st.number_input(
            "Tolerancia (céntimos)",
            min_value=0,
            max_value=10000,
            value=0,
            step=1,
            help="Margen permitido en céntimos para cuadrar pagos con facturas 90. 0 = exacto. Ej: 500 permite diferencias de hasta 5,00 €."
        )
    tolerancia_euros = tolerancia_centimos / 100.0

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📅 Día seleccionado: **{fecha_seleccionada.strftime('%d/%m/%Y')}** ({len(df_pagos)} pagos)")
    with col2:
        ejecutar_cruce = st.button("🔄 Ejecutar Cruce", type="primary", use_container_width=True)
    
    # Solo ejecutar si se pulsa el botón
    if ejecutar_cruce:
        st.info("""
            ℹ️ **Filtros aplicados:**
            - ✅ Base = TODAS las facturas 90 de COBRA (TSS)
            - ✅ CASO 1: 90 con match en PRISMA → socios TDE/TME de PRISMA + TSOL si hay diferencia
            - ✅ CASO 2: 90 sin match en PRISMA → busca UNA factura TSOL por CIF grupal
            - ✅ Facturas 90 con fecha emisión ≤ fecha del pago
            - ✅ Solo facturas 90 con importe ≤ importe del pago
            - ⛔ Facturas con importe negativo se ignoran
            """)
        
        with st.spinner("⏳ Buscando combinaciones óptimas de facturas... esto puede tardar unos segundos"):
            inicio = time.time()
            
            # Normalizar CIF_UTE solo cuando se va a ejecutar el cruce
            df_pagos_normalizado = df_pagos.copy()
            df_pagos_normalizado['CIF_UTE'] = (
                df_pagos_normalizado['CIF_UTE']
                .astype(str)
                .str.replace(".0", "", regex=False)
                .str.strip()
                .str.upper()
            )

            df_resultados = cruzar_pagos_con_prisma_exacto(
                df_pagos_normalizado,
                df_prisma_90,
                df_prisma,
                col_num_factura_prisma,
                tolerancia_euros,
                maestro_map=st.session_state.get('maestro_map', {})
            )
            
            fin = time.time()
            
            # Guardar resultados en session_state
            st.session_state.df_resultados = df_resultados
            st.session_state.fecha_resultados = fecha_seleccionada
            st.session_state.df_pagos_normalizado = df_pagos_normalizado
            
            st.success(f"✅ Cruce completado en {fin - inicio:.2f} segundos")
    
    # -------------------------------
    # 5️⃣ MOSTRAR RESULTADOS SI EXISTEN
    # -------------------------------
    
    if "df_resultados" in st.session_state and st.session_state.df_resultados is not None:
        df_resultados = st.session_state.df_resultados
        
        st.markdown("---")
        st.subheader("📊 Resultados del cruce")
        
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        
        total_pagos         = len(df_resultados)
        pagos_con_facturas  = df_resultados['facturas_90_asignadas'].notna().sum()
        pagos_sin_facturas  = total_pagos - pagos_con_facturas
        pagos_con_advertencia = df_resultados['advertencia'].notna().sum()
        importe_total_pagos = df_resultados['importe_pago'].sum()
        importe_total_90    = df_resultados['importe_facturas_90'].sum()
        diferencia_pago_vs_90 = df_resultados['diferencia_pago_vs_90'].sum()
        
        # Calcular totales de socios desde el desglose
        importe_total_socios         = 0.0
        diferencia_total_90_vs_socios = 0.0
        caso1_count = 0
        caso2_count = 0
        caso2_tsol_ok = 0

        for _, row in df_resultados.iterrows():
            if row['desglose_facturas_90'] is not None:
                for factura_90 in row['desglose_facturas_90']:
                    importe_total_socios          += factura_90['importe_socios']
                    diferencia_total_90_vs_socios += factura_90['diferencia_90_socios']
                    if factura_90.get('caso', '').startswith('CASO 1'):
                        caso1_count += 1
                    else:
                        caso2_count += 1
                        if factura_90.get('estado_cobra', '').startswith('✅'):
                            caso2_tsol_ok += 1

        with col1:
            st.metric("Total Pagos", total_pagos)
        with col2:
            st.metric("Con Facturas 90", pagos_con_facturas, delta=f"{(pagos_con_facturas/total_pagos*100):.1f}%")
        with col3:
            st.metric("Sin Facturas", pagos_sin_facturas)
        with col4:
            if pagos_con_advertencia > 0:
                st.metric("⚠️ Con Advertencia", pagos_con_advertencia, delta="Revisar", delta_color="inverse")
            else:
                st.metric("Dif. Pago vs 90", f"{diferencia_pago_vs_90:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
        
        # Métricas adicionales de importes
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Total Pagos", f"{importe_total_pagos:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
        with col2:
            st.metric("🔵 Facturas 90", f"{importe_total_90:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
        with col3:
            st.metric("🟢 Facturas Socios", f"{importe_total_socios:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
        with col4:
            st.metric("⚠️ Dif. 90 vs Socios", f"{diferencia_total_90_vs_socios:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

        # Resumen CASO 1 / CASO 2
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📘 CASO 1 (con PRISMA)", caso1_count)
        with col2:
            st.metric("📙 CASO 2 (sin PRISMA)", caso2_count)
        with col3:
            st.metric("✅ CASO 2 con TSOL encontrada", caso2_tsol_ok, delta=f"{(caso2_tsol_ok/caso2_count*100):.1f}%" if caso2_count else "0%")
        
        if pagos_con_advertencia > 0:
            st.warning(f"⚠️ **{pagos_con_advertencia} pago(s) tienen advertencia**: Revisa la columna 'advertencia' en el Excel.")
        
        st.dataframe(df_resultados, use_container_width=True, height=400)
        
        # Vista detallada con desglose
        with st.expander("🔍 Ver desglose detallado de facturas por pago"):
            for idx, row in df_resultados.iterrows():
                if pd.notna(row['facturas_90_asignadas']) and row['desglose_facturas_90'] is not None:
                    st.markdown(f"### 💰 Pago {idx+1}: {row['importe_pago']:,.2f} € ({row['CIF_UTE']})".replace(",", "X").replace(".", ",").replace("X", "."))
                    st.markdown(f"**📅 Fecha:** {row['fecha_pago'].strftime('%d/%m/%Y')}")
                    st.markdown(f"**🔵 Total facturas 90:** {row['importe_facturas_90']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                    st.markdown(f"**⚠️ Diferencia Pago vs 90:** {row['diferencia_pago_vs_90']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                    
                    if pd.notna(row['advertencia']):
                        st.warning(row['advertencia'])
                    
                    st.markdown("---")
                    
                    for i, factura_90_data in enumerate(row['desglose_facturas_90'], 1):
                        caso_label = factura_90_data.get('caso', '')
                        cif_cliente = factura_90_data.get('cif_cliente_final', '')
                        st.markdown(f"#### 📄 Factura 90 #{i}: {factura_90_data['factura_90']} — {caso_label}")
                        st.markdown(f"**Importe factura 90:** {factura_90_data['importe_90']:,.2f} €  |  **CIF cliente final:** `{cif_cliente}`".replace(",", "X").replace(".", ",").replace("X", "."))
                        
                        # Socios de PRISMA (solo CASO 1)
                        socios_prisma = factura_90_data.get('socios_prisma', [])
                        importe_prisma = factura_90_data.get('importe_socios_prisma', 0)
                        if socios_prisma:
                            st.markdown("**🔵 Socios en PRISMA:**")
                            for socio in socios_prisma:
                                st.markdown(f"  • {socio['num_factura']} ({socio['cif']}): {socio['importe']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                            st.markdown(f"**Total socios PRISMA:** {importe_prisma:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                        elif caso_label.startswith('CASO 1'):
                            st.markdown("**🔵 Socios PRISMA:** Sin socios encontrados")
                        
                        # Socios de COBRA (TSOL)
                        socios_cobra = factura_90_data.get('socios_cobra', [])
                        importe_cobra = factura_90_data.get('importe_socios_cobra', 0)
                        estado_cobra = factura_90_data.get('estado_cobra', None)
                        
                        if estado_cobra:
                            if socios_cobra:
                                st.markdown("**🟠 Facturas COBRA (TSOL):**")
                                for socio in socios_cobra:
                                    st.markdown(f"  • {socio['num_factura']} ({socio['fuente']}): {socio['importe']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                                st.markdown(f"**Total TSOL:** {importe_cobra:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                            st.markdown(f"**Estado COBRA:** {estado_cobra}")
                        
                        diferencia_color = "🔴" if abs(factura_90_data['diferencia_90_socios']) > 0.01 else "✅"
                        st.markdown(f"{diferencia_color} **Diferencia 90 vs Socios:** {factura_90_data['diferencia_90_socios']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                        
                        if i < len(row['desglose_facturas_90']):
                            st.markdown("---")
                    
                    st.markdown("---")
                    st.markdown("---")
        
        # -------------------------------
        # 6️⃣ DESCARGAR EXCEL CON DESGLOSE
        # -------------------------------
        
        cif_a_nombre = {}
        if "df_pagos_normalizado" in st.session_state and 'denominacion' in st.session_state.df_pagos_normalizado.columns:
            for _, pago in st.session_state.df_pagos_normalizado.iterrows():
                cif = pago['CIF_UTE']
                nombre = pago.get('denominacion', 'DESCONOCIDO')
                if pd.notna(nombre):
                    cif_a_nombre[cif] = str(nombre)
        
        filas_excel = []
        for _, row in df_resultados.iterrows():
            nombre_ute = cif_a_nombre.get(row['CIF_UTE'], 'DESCONOCIDO')
            
            if row['desglose_facturas_90'] is not None:
                for f90 in row['desglose_facturas_90']:
                    socios_prisma = f90.get('socios_prisma', [])
                    socios_prisma_str = ' | '.join([
                        f"{s['num_factura']} ({s['cif']}): {s['importe']:.2f}€"
                        for s in socios_prisma
                    ]) if socios_prisma else "Sin socios en PRISMA"
                    
                    socios_cobra = f90.get('socios_cobra', [])
                    socios_cobra_str = ' | '.join([
                        f"{s['num_factura']} ({s['fuente']}): {s['importe']:.2f}€"
                        for s in socios_cobra
                    ]) if socios_cobra else ""
                    
                    estado_cobra = f90.get('estado_cobra', '')
                    
                    filas_excel.append({
                        'CIF_UTE':                row['CIF_UTE'],
                        'Nombre_UTE':             nombre_ute,
                        'Fecha_Pago':             row['fecha_pago'].date() if pd.notna(row['fecha_pago']) else None,
                        'Importe_Pago':           row['importe_pago'],
                        'Factura_90':             f90['factura_90'],
                        'Caso':                   f90.get('caso', ''),
                        'CIF_Cliente_Final':      f90.get('cif_cliente_final', ''),
                        'Importe_90':             f90['importe_90'],
                        'Socios_PRISMA':          socios_prisma_str,
                        'Importe_Socios_PRISMA':  f90.get('importe_socios_prisma', 0),
                        'Socios_COBRA_TSOL':      socios_cobra_str if socios_cobra_str else ('N/A' if not estado_cobra else ''),
                        'Importe_Socios_COBRA':   f90.get('importe_socios_cobra', 0),
                        'Estado_COBRA':           estado_cobra if estado_cobra else 'Sin diferencia',
                        'Total_Socios':           f90['importe_socios'],
                        'Diferencia_90_vs_Socios': f90['diferencia_90_socios'],
                        'Diferencia_Pago_vs_90':  row['diferencia_pago_vs_90'],
                        'Advertencia':            row['advertencia'] if pd.notna(row['advertencia']) else ''
                    })
            else:
                filas_excel.append({
                    'CIF_UTE':                row['CIF_UTE'],
                    'Nombre_UTE':             nombre_ute,
                    'Fecha_Pago':             row['fecha_pago'].date() if pd.notna(row['fecha_pago']) else None,
                    'Importe_Pago':           row['importe_pago'],
                    'Factura_90':             None,
                    'Caso':                   '',
                    'CIF_Cliente_Final':      '',
                    'Importe_90':             0.0,
                    'Socios_PRISMA':          None,
                    'Importe_Socios_PRISMA':  0.0,
                    'Socios_COBRA_TSOL':      None,
                    'Importe_Socios_COBRA':   0.0,
                    'Estado_COBRA':           '',
                    'Total_Socios':           0.0,
                    'Diferencia_90_vs_Socios': 0.0,
                    'Diferencia_Pago_vs_90':  row['diferencia_pago_vs_90'],
                    'Advertencia':            row['advertencia'] if pd.notna(row['advertencia']) else ''
                })
        
        df_excel = pd.DataFrame(filas_excel)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_excel.to_excel(writer, index=False, sheet_name="Desglose_Detallado")
        output.seek(0)

        st.download_button(
            label="📥 Descargar resultados en Excel",
            data=output,
            file_name=f"resultados_cruce_{st.session_state.fecha_resultados.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    elif ejecutar_cruce is False and "df_resultados" not in st.session_state:
        st.info("👆 Pulsa el botón 'Ejecutar Cruce' para iniciar el proceso")

