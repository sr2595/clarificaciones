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
    """
    Aplica el impuesto correspondiente a cada fila de PRISMA
    y devuelve el DataFrame con nueva columna 'IMPORTE_CON_IMPUESTO'.
    """
    factores = {
        "IGIC - 7": 1.07,
        "IPSIC - 10": 1.10,
        "IPSIM - 8": 1.08,
        "IVA - 0": 1.00,
        "IVA - 21": 1.21,
        "EXENTO": 1.0,
        "IVA - EXENTO": 1.0,
    }

    # Normalizamos la columna tipo impuesto
    df_prisma[col_tipo_impuesto] = df_prisma[col_tipo_impuesto].astype(str).str.strip().str.upper()

    # Crear nueva columna con el importe ya con impuesto aplicado
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

# --------- 2) Subida y normalización de COBRA ---------
archivo_cobra = st.file_uploader("Sube el archivo Excel DetalleDocumentos de Cobra", type=["xlsx", "xls"])

# Guardar bytes en session_state
if archivo_cobra is not None:
    st.session_state.cobra_bytes = archivo_cobra.getvalue()

# Si no hay bytes, no seguimos
if "cobra_bytes" not in st.session_state:
    st.stop()

# PROCESAR COBRA SOLO UNA VEZ
if "df_cobra_procesado" not in st.session_state:
    st.info("⏳ Procesando archivo COBRA por primera vez...")
    
    # Leer COBRA desde bytes
    df_raw = pd.read_excel(BytesIO(st.session_state.cobra_bytes), header=None)

    # Buscar fila que contiene la cabecera
    header_row = None
    for i in range(min(20, len(df_raw))):
        vals = [str(x).lower() for x in df_raw.iloc[i].tolist()]
        if any("factura" in v or "fecha" in v or "importe" in v for v in vals):
            header_row = i
            break

    if header_row is None:
        st.error("❌ No se encontró cabecera reconocible en el archivo Excel")
        st.stop()

    # Releer usando esa fila como cabecera
    df = pd.read_excel(BytesIO(st.session_state.cobra_bytes), header=header_row)

    # --- Detectar columnas ---
    col_fecha_emision = find_col(df, ['FECHA', 'Fecha Emision', 'Fecha Emisión', 'FX_EMISION'])
    col_factura       = find_col(df, ['FACTURA', 'Nº Factura', 'NRO_FACTURA', 'Núm.Doc.Deuda'])
    col_importe       = find_col(df, ['IMPORTE', 'TOTAL', 'TOTAL_FACTURA'])
    col_cif           = find_col(df, ['T.Doc. - Núm.Doc.', 'CIF', 'NIF', 'CIF_CLIENTE', 'NIF_CLIENTE'])
    col_nombre_cliente= find_col(df, ['NOMBRE', 'CLIENTE', 'RAZON_SOCIAL'])
    col_sociedad      = find_col(df, ['SOCIEDAD', 'Sociedad', 'SOC', 'EMPRESA'])
    col_grupo         = find_col(df, ['CIF_GRUPO', 'GRUPO', 'CIF Grupo'])
    col_nombre_grupo  = find_col(df, ['Nombre Grupo', 'GRUPO_NOMBRE', 'RAZON_SOCIAL_GRUPO'])

    faltan = []
    if not col_fecha_emision: faltan.append("fecha emisión")
    if not col_factura:       faltan.append("nº factura")
    if not col_importe:       faltan.append("importe")
    if not col_cif:           faltan.append("CIF")
    if not col_grupo:         faltan.append("CIF grupo")
    if not col_nombre_grupo:  faltan.append("Nombre grupo")

    if faltan:
        st.error("❌ No se pudieron localizar estas columnas: " + ", ".join(faltan))
        st.stop()

    # --- Normalizar ---
    df[col_fecha_emision] = pd.to_datetime(df[col_fecha_emision], dayfirst=True, errors='coerce')
    df[col_factura] = df[col_factura].astype(str)
    df[col_cif] = df[col_cif].astype(str)
    df['IMPORTE_CORRECTO'] = df[col_importe].apply(convertir_importe_europeo)
    df['IMPORTE_CENT'] = (df['IMPORTE_CORRECTO'] * 100).round().astype("Int64")

    # Detectar UTES
    df['ES_UTE'] = df[col_cif].astype(str).str.replace(" ", "").str.contains(r"L-00U")
    
    # Guardar en session_state
    st.session_state.df_cobra_procesado = df
    st.session_state.col_fecha_emision = col_fecha_emision
    st.session_state.col_factura = col_factura
    st.session_state.col_importe = col_importe
    st.session_state.col_cif = col_cif
    
    # Resumen del archivo
    total = df['IMPORTE_CORRECTO'].sum(skipna=True)
    minimo = df['IMPORTE_CORRECTO'].min(skipna=True)
    maximo = df['IMPORTE_CORRECTO'].max(skipna=True)

    st.write("**📊 Resumen del archivo COBRA:**")
    st.write(f"- Número total de facturas: {len(df)}")
    st.write(f"- Suma total importes: {total:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe mínimo: {minimo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe máximo: {maximo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
else:
    # Recuperar desde session_state
    df = st.session_state.df_cobra_procesado
    col_fecha_emision = st.session_state.col_fecha_emision
    col_factura = st.session_state.col_factura
    col_importe = st.session_state.col_importe
    col_cif = st.session_state.col_cif
    
    st.success(f"✅ Archivo COBRA ya cargado ({len(df)} filas)")

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
if "cobros_bytes" not in st.session_state:
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
        'CIF_UTE': ['cif', 'cif_ute']
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

    df_pagos = df_cobros_filtrado[columnas_cruce].copy()

    st.subheader("🔍 Pagos filtrados para cruce")
    st.dataframe(df_pagos.head(10), use_container_width=True)
    st.write(f"Total importes en el día: {df_pagos['importe'].sum():,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

    #######--- 5) PREPARAR DATOS PARA CRUCE (SOLO UNA VEZ) ---#######

    # PREPARAR df_prisma_90 SOLO UNA VEZ y guardarlo en session_state
    if "df_prisma_90_preparado" not in st.session_state:
        st.info("⏳ Preparando datos de PRISMA para cruce (solo la primera vez)...")
        
        # NORMALIZACIONES BASE
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

        # -------------------------------
        # 1️⃣ OBTENER CIF UTE POR Id UTE (desde socios)
        # -------------------------------

        df_temp = df_prisma.copy()
        df_temp[col_num_factura_prisma] = df_temp[col_num_factura_prisma].astype(str).str.strip()
        df_temp['Id UTE'] = df_temp['Id UTE'].astype(str).str.strip()
        df_temp['CIF'] = df_temp['CIF'].astype(str).str.strip()

        # Facturas que NO empiezan por 90
        df_sin_90 = df_temp[~df_temp[col_num_factura_prisma].str.startswith("90")].copy()
        cif_por_ute = df_sin_90.groupby('Id UTE')['CIF'].first().to_dict()

        # -------------------------------
        # 2️⃣ FACTURAS 90 + CIF UTE REAL
        # -------------------------------

        df_prisma_90 = df_temp[df_temp[col_num_factura_prisma].str.startswith("90")].copy()
        df_prisma_90['Id UTE'] = df_prisma_90['Id UTE'].astype(str).str.strip()
        df_prisma_90['CIF_UTE_REAL'] = df_prisma_90['Id UTE'].apply(lambda x: cif_por_ute.get(x, "NONE"))

        # Añadir columnas de nombre de UTE y cliente final si existen en df_prisma
        if 'Nombre UTE' in df_prisma.columns:
            df_prisma_90['Nombre_UTE'] = df_prisma_90['Id UTE'].map(df_prisma.set_index('Id UTE')['Nombre UTE'])
        else:
            df_prisma_90['Nombre_UTE'] = "DESCONOCIDO"

        if 'Nombre Cliente' in df_prisma.columns:
            df_prisma_90['Nombre_Cliente'] = df_prisma_90['CIF_UTE_REAL'].map(df_prisma.set_index('CIF')['Nombre Cliente'])
        else:
            df_prisma_90['Nombre_Cliente'] = "DESCONOCIDO"
        
        # Guardar en session_state
        st.session_state.df_prisma_90_preparado = df_prisma_90
        st.success("✅ Datos de PRISMA preparados")
    else:
        # Recuperar desde session_state
        df_prisma_90 = st.session_state.df_prisma_90_preparado

    # DEBUG mínimo
    with st.expander("🔍 Info facturas 90 preparadas"):
        st.write("ℹ️ Filas de facturas 90:", len(st.session_state.df_prisma_90_preparado))
        st.write("ℹ️ Facturas 90 sin CIF_UTE_REAL asignado:", (st.session_state.df_prisma_90_preparado['CIF_UTE_REAL'] == "NONE").sum())
        st.dataframe(st.session_state.df_prisma_90_preparado[[col_num_factura_prisma, 'Id UTE', 'CIF_UTE_REAL', 'Nombre_UTE', 'Nombre_Cliente']].head(20))
    
    # Usar el df_prisma_90 desde session_state
    df_prisma_90 = st.session_state.df_prisma_90_preparado

    # -------------------------------
    # 3️⃣ FUNCIÓN OR-TOOLS CON SOCIOS
    # -------------------------------
    def cruzar_pagos_con_prisma_exacto(df_pagos, df_prisma_90, df_prisma_completo, col_num_factura_prisma, tolerancia=0.01):
        """
        Cruza pagos con facturas 90 y además busca las facturas de socios asociadas a cada factura 90
        """
        resultados = []
        facturas_por_cif = {cif: g.copy() for cif, g in df_prisma_90.groupby('CIF_UTE_REAL')}
        
        # Crear diccionario de facturas de socios por Id UTE (facturas que NO son 90)
        df_socios = df_prisma_completo[~df_prisma_completo[col_num_factura_prisma].astype(str).str.startswith("90")].copy()
        socios_por_ute = {}
        for id_ute, grupo in df_socios.groupby('Id UTE'):
            socios_por_ute[str(id_ute).strip()] = grupo[[col_num_factura_prisma, 'IMPORTE_CORRECTO', 'CIF']].copy()

        for idx, pago in df_pagos.iterrows():
            try:
                cif_pago = pago['CIF_UTE']
                importe_pago = pago['importe']
                fecha_pago = pago['fec_operacion']

                if cif_pago not in facturas_por_cif:
                    resultados.append({
                        'CIF_UTE': cif_pago,
                        'fecha_pago': fecha_pago,
                        'importe_pago': importe_pago,
                        'facturas_90_asignadas': None,
                        'importe_facturas_90': 0.0,
                        'desglose_facturas_90': None,
                        'diferencia_pago_vs_90': importe_pago
                    })
                    continue

                df_facturas = facturas_por_cif[cif_pago].sort_values('IMPORTE_CORRECTO', ascending=True)
                numeros_facturas = df_facturas[col_num_factura_prisma].tolist()
                importes_facturas = df_facturas['IMPORTE_CORRECTO'].tolist()
                ids_ute = df_facturas['Id UTE'].tolist()

                # --- Modelo OR-Tools para facturas 90
                model = cp_model.CpModel()
                n = len(importes_facturas)
                pagos_cent = int(round(importe_pago * 100))
                facturas_cent = [int(round(f * 100)) for f in importes_facturas]
                tol_cent = int(round(tolerancia * 100))

                x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
                model.Add(sum(x[i] * facturas_cent[i] for i in range(n)) >= pagos_cent - tol_cent)
                model.Add(sum(x[i] * facturas_cent[i] for i in range(n)) <= pagos_cent + tol_cent)

                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 3
                solver.parameters.log_search_progress = False
                status = solver.Solve(model)

                if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                    seleccion = [i for i in range(n) if solver.Value(x[i]) == 1]
                    
                    # Desglose detallado por cada factura 90
                    desglose_por_factura_90 = []
                    importe_facturas_90 = 0.0
                    
                    for i in seleccion:
                        num_factura_90 = numeros_facturas[i]
                        importe_factura_90 = importes_facturas[i]
                        id_ute = str(ids_ute[i]).strip()
                        
                        importe_facturas_90 += importe_factura_90
                        
                        # Buscar socios de esta factura 90
                        socios_lista = []
                        importe_socios_de_esta_90 = 0.0
                        
                        if id_ute in socios_por_ute:
                            df_socios_ute = socios_por_ute[id_ute]
                            for _, socio in df_socios_ute.iterrows():
                                num_factura_socio = str(socio[col_num_factura_prisma])
                                importe_socio = socio['IMPORTE_CORRECTO']
                                cif_socio = str(socio['CIF'])
                                socios_lista.append({
                                    'num_factura': num_factura_socio,
                                    'cif': cif_socio,
                                    'importe': importe_socio
                                })
                                importe_socios_de_esta_90 += importe_socio
                        
                        # Diferencia entre esta factura 90 y sus socios
                        diferencia_90_socios = importe_factura_90 - importe_socios_de_esta_90
                        
                        desglose_por_factura_90.append({
                            'factura_90': num_factura_90,
                            'importe_90': importe_factura_90,
                            'socios': socios_lista,
                            'importe_socios': importe_socios_de_esta_90,
                            'diferencia_90_socios': diferencia_90_socios
                        })
                    
                    # String resumen de facturas 90 para visualización rápida
                    facturas_90_str = ', '.join([d['factura_90'] for d in desglose_por_factura_90])
                    
                else:
                    facturas_90_str = None
                    importe_facturas_90 = 0.0
                    desglose_por_factura_90 = None

                # Diferencia pago vs facturas 90
                diferencia_pago_vs_90 = importe_pago - importe_facturas_90

                resultados.append({
                    'CIF_UTE': cif_pago,
                    'fecha_pago': fecha_pago,
                    'importe_pago': importe_pago,
                    'facturas_90_asignadas': facturas_90_str,
                    'importe_facturas_90': importe_facturas_90,
                    'desglose_facturas_90': desglose_por_factura_90,  # Aquí guardamos el desglose completo
                    'diferencia_pago_vs_90': diferencia_pago_vs_90
                })
            
            except Exception as e:
                # Si falla un pago individual, continuar con el siguiente
                resultados.append({
                    'CIF_UTE': pago.get('CIF_UTE', 'ERROR'),
                    'fecha_pago': pago.get('fec_operacion'),
                    'importe_pago': pago.get('importe', 0),
                    'facturas_90_asignadas': f"ERROR: {str(e)}",
                    'importe_facturas_90': 0.0,
                    'desglose_facturas_90': None,
                    'diferencia_pago_vs_90': pago.get('importe', 0)
                })
                continue

        return pd.DataFrame(resultados)

    # -------------------------------
    # 4️⃣ BOTÓN PARA EJECUTAR EL SOLVER
    # -------------------------------
    
    st.markdown("---")
    st.subheader("🚀 Ejecutar cruce de pagos con facturas")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📅 Día seleccionado: **{fecha_seleccionada.strftime('%d/%m/%Y')}** ({len(df_pagos)} pagos)")
    with col2:
        ejecutar_cruce = st.button("🔄 Ejecutar Cruce", type="primary", use_container_width=True)
    
    # Solo ejecutar si se pulsa el botón
    if ejecutar_cruce:
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
                df_prisma,  # Pasamos el df_prisma completo para acceder a facturas de socios
                col_num_factura_prisma,
                0.01
            )
            
            fin = time.time()
            
            # Guardar resultados en session_state
            st.session_state.df_resultados = df_resultados
            st.session_state.fecha_resultados = fecha_seleccionada
            
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
        
        total_pagos = len(df_resultados)
        pagos_con_facturas = df_resultados['facturas_90_asignadas'].notna().sum()
        pagos_sin_facturas = total_pagos - pagos_con_facturas
        importe_total_pagos = df_resultados['importe_pago'].sum()
        importe_total_90 = df_resultados['importe_facturas_90'].sum()
        diferencia_pago_vs_90 = df_resultados['diferencia_pago_vs_90'].sum()
        
        # Calcular total de socios y diferencias desde el desglose
        importe_total_socios = 0.0
        diferencia_total_90_vs_socios = 0.0
        for _, row in df_resultados.iterrows():
            if row['desglose_facturas_90'] is not None:
                for factura_90 in row['desglose_facturas_90']:
                    importe_total_socios += factura_90['importe_socios']
                    diferencia_total_90_vs_socios += factura_90['diferencia_90_socios']
        
        with col1:
            st.metric("Total Pagos", total_pagos)
        with col2:
            st.metric("Con Facturas 90", pagos_con_facturas, delta=f"{(pagos_con_facturas/total_pagos*100):.1f}%")
        with col3:
            st.metric("Sin Facturas", pagos_sin_facturas)
        with col4:
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
        
        # Tabla de resultados
        st.dataframe(df_resultados, use_container_width=True, height=400)
        
        # Vista detallada con desglose
        with st.expander("🔍 Ver desglose detallado de facturas por pago"):
            for idx, row in df_resultados.iterrows():
                if pd.notna(row['facturas_90_asignadas']) and row['desglose_facturas_90'] is not None:
                    st.markdown(f"### 💰 Pago {idx+1}: {row['importe_pago']:,.2f} € ({row['CIF_UTE']})".replace(",", "X").replace(".", ",").replace("X", "."))
                    st.markdown(f"**📅 Fecha:** {row['fecha_pago'].strftime('%d/%m/%Y')}")
                    st.markdown(f"**🔵 Total facturas 90:** {row['importe_facturas_90']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                    st.markdown(f"**⚠️ Diferencia Pago vs 90:** {row['diferencia_pago_vs_90']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                    
                    st.markdown("---")
                    
                    # Mostrar cada factura 90 individualmente con sus socios
                    for i, factura_90_data in enumerate(row['desglose_facturas_90'], 1):
                        st.markdown(f"#### 📄 Factura 90 #{i}: {factura_90_data['factura_90']}")
                        st.markdown(f"**Importe factura 90:** {factura_90_data['importe_90']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                        
                        # Mostrar socios de esta factura 90
                        if factura_90_data['socios']:
                            st.markdown("**🟢 Facturas de socios que la componen:**")
                            for socio in factura_90_data['socios']:
                                st.markdown(f"  • {socio['num_factura']} ({socio['cif']}): {socio['importe']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                            st.markdown(f"**Total socios:** {factura_90_data['importe_socios']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                        else:
                            st.markdown("**🟢 Facturas de socios:** Sin socios encontrados")
                            st.markdown(f"**Total socios:** 0,00 €")
                        
                        # Diferencia de esta factura 90 específica
                        diferencia_color = "🔴" if factura_90_data['diferencia_90_socios'] != 0 else "✅"
                        st.markdown(f"{diferencia_color} **Diferencia 90 vs Socios:** {factura_90_data['diferencia_90_socios']:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                        
                        if i < len(row['desglose_facturas_90']):
                            st.markdown("---")
                    
                    st.markdown("---")
                    st.markdown("---")
        
        # -------------------------------
        # 6️⃣ DESCARGAR EXCEL CON DESGLOSE
        # -------------------------------
        
        # Crear DataFrame plano con desglose de cada factura 90
        filas_excel = []
        for _, row in df_resultados.iterrows():
            if row['desglose_facturas_90'] is not None:
                for factura_90_data in row['desglose_facturas_90']:
                    # Convertir socios a string
                    socios_str = ' | '.join([
                        f"{s['num_factura']} ({s['cif']}): {s['importe']:.2f}€" 
                        for s in factura_90_data['socios']
                    ]) if factura_90_data['socios'] else "Sin socios"
                    
                    filas_excel.append({
                        'CIF_UTE': row['CIF_UTE'],
                        'Fecha_Pago': row['fecha_pago'],
                        'Importe_Pago': row['importe_pago'],
                        'Factura_90': factura_90_data['factura_90'],
                        'Importe_90': factura_90_data['importe_90'],
                        'Facturas_Socios': socios_str,
                        'Importe_Socios': factura_90_data['importe_socios'],
                        'Diferencia_90_vs_Socios': factura_90_data['diferencia_90_socios'],
                        'Diferencia_Pago_vs_90': row['diferencia_pago_vs_90']
                    })
            else:
                # Pago sin facturas asignadas
                filas_excel.append({
                    'CIF_UTE': row['CIF_UTE'],
                    'Fecha_Pago': row['fecha_pago'],
                    'Importe_Pago': row['importe_pago'],
                    'Factura_90': None,
                    'Importe_90': 0.0,
                    'Facturas_Socios': None,
                    'Importe_Socios': 0.0,
                    'Diferencia_90_vs_Socios': 0.0,
                    'Diferencia_Pago_vs_90': row['diferencia_pago_vs_90']
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
