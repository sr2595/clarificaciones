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

def filtrar_cobra_ligero(df, col_sociedad, col_importe):
    """
    Filtra COBRA para quedarse solo con lo necesario:
    - Sociedades TSS y TSOL
    - Importes positivos
    """
    filas_originales = len(df)
    
    # 1️⃣ Filtrar por sociedad TSS y TSOL
    if col_sociedad:
        soc_norm = df[col_sociedad].astype(str).str.strip().str.upper()
        df = df[soc_norm.isin(['TSS', 'TSOL'])].copy()
    
    # 2️⃣ Eliminar importes negativos y cero
    if col_importe:
        importes = df[col_importe].apply(convertir_importe_europeo)
        df = df[importes > 0].copy()
    
    filas_finales = len(df)
    reduccion = (1 - filas_finales / filas_originales) * 100 if filas_originales > 0 else 0
    
    st.success(f"✂️ COBRA reducido: {filas_originales:,} → {filas_finales:,} filas ({reduccion:.1f}% eliminado)")
    
    return df

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

# --------- 2) Subida y normalización de COBRA ---------
archivo_cobra = st.file_uploader("Sube el archivo Excel DetalleDocumentos de Cobra", type=["xlsx", "xls", "csv"])

# Guardar bytes en session_state
if archivo_cobra is not None:
    st.session_state.cobra_bytes = archivo_cobra.getvalue()
    st.session_state.cobra_nombre = archivo_cobra.name  

# Si no hay bytes, no seguimos
if "cobra_bytes" not in st.session_state and "df_cobra_procesado" not in st.session_state:
    st.stop()

if "df_cobra_procesado" not in st.session_state:
    st.info("⏳ Procesando archivo COBRA por primera vez...")
    
    nombre = st.session_state.get('cobra_nombre', '')
    cobra_bytes = st.session_state.cobra_bytes

    # --- Leer archivo ---
    if nombre.endswith('.csv'):
        chunks = []
        for chunk in pd.read_csv(
            BytesIO(cobra_bytes),
            sep=";",
            encoding="latin1",
            on_bad_lines="skip",
            chunksize=50_000
        ):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        del chunks
    else:
        # Detectar cabecera solo leyendo 20 filas
        df_raw = pd.read_excel(BytesIO(cobra_bytes), header=None, engine="openpyxl", nrows=20)
        header_row = None
        for i in range(len(df_raw)):
            vals = [str(x).lower() for x in df_raw.iloc[i].tolist()]
            if any("factura" in v or "fecha" in v or "importe" in v for v in vals):
                header_row = i
                break
        del df_raw  # 🧹 liberar
        
        if header_row is None:
            st.error("❌ No se encontró cabecera reconocible en el archivo Excel")
            st.stop()
        
        df = pd.read_excel(BytesIO(cobra_bytes), header=header_row, engine="openpyxl")
    
    # --- Detectar columnas ANTES de filtrar ---
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

    # 🔥 FILTRAR COBRA AQUÍ — antes de normalizar y guardar en session_state
    df = filtrar_cobra_ligero(df, col_sociedad, col_importe)
    
    # 🧹 Liberar bytes crudos — ya no los necesitamos
    del st.session_state.cobra_bytes

    # --- Normalizar ---
    df[col_fecha_emision] = pd.to_datetime(df[col_fecha_emision], dayfirst=True, errors='coerce')
    df[col_factura] = df[col_factura].astype(str)
    df[col_cif] = df[col_cif].astype(str).str.strip()  # conservar original (con L-00)
    if col_grupo:
        df[col_grupo] = df[col_grupo].astype(str).str.strip()
    df['IMPORTE_CORRECTO'] = df[col_importe].apply(convertir_importe_europeo)
    df['IMPORTE_CENT'] = (df['IMPORTE_CORRECTO'] * 100).round().astype("Int64")
    df['ES_UTE'] = df[col_cif].astype(str).str.replace(" ", "").str.contains(r"L-00U")

    # Guardar en session_state
    st.session_state.df_cobra_procesado = df
    st.session_state.col_fecha_emision = col_fecha_emision
    st.session_state.col_factura = col_factura
    st.session_state.col_importe = col_importe
    st.session_state.col_cif = col_cif
    st.session_state.col_sociedad = col_sociedad
    st.session_state.col_grupo = col_grupo

    # ✅ Calcular resumen DESPUÉS de normalizar
    total  = df['IMPORTE_CORRECTO'].sum(skipna=True)
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
    col_sociedad = st.session_state.get('col_sociedad', None)
    col_grupo = st.session_state.get('col_grupo', None)
    
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

    # PASO B: Base = todas las facturas 90 de COBRA (TSS)
    if "df_prisma_90_preparado" not in st.session_state:
        st.info("🔄 Preparando facturas 90 desde COBRA como base...")

        df_cobra_cruce = df.copy()
        df_cobra_cruce['Num_Factura_Norm'] = df_cobra_cruce[col_factura].astype(str).str.strip().str.upper()
        df_cobra_cruce['CIF_Norm'] = (
            df_cobra_cruce[col_cif]
            .astype(str)
            .str.replace(" ", "")
            .str.strip()
            .str.upper()
            .str.replace("L-00", "", regex=False)
        )
        # CIF original sin normalizar (para buscar en TSOL)
        df_cobra_cruce['CIF_ORIGINAL'] = df_cobra_cruce[col_cif].astype(str).str.strip()

        if col_sociedad:
            df_cobra_cruce['Sociedad_Norm'] = df_cobra_cruce[col_sociedad].astype(str).str.strip().str.upper()
            df_cobra_tss = df_cobra_cruce[df_cobra_cruce['Sociedad_Norm'] == 'TSS'].copy()
        else:
            df_cobra_tss = df_cobra_cruce.copy()
            st.warning("⚠️ No se detectó columna 'Sociedad'. Se usarán todas las facturas.")

        df_cobra_90 = df_cobra_tss[
            df_cobra_tss['Num_Factura_Norm'].str.startswith('90')
        ].copy()

        st.write(f"📊 Facturas 90 en COBRA (TSS): **{len(df_cobra_90)}**")

        df_prisma_90_base['Num_Factura_Norm_P'] = df_prisma_90_base['Num_Factura_Norm']

        df_prisma_90 = pd.merge(
            df_cobra_90[['Num_Factura_Norm', 'CIF_Norm', 'CIF_ORIGINAL']].drop_duplicates(subset='Num_Factura_Norm'),
            df_prisma_90_base,
            on='Num_Factura_Norm',
            how='left'
        )

        # Para las sin match en PRISMA: CIF_UTE_REAL = CIF_Norm (CIF de COBRA sin L-00)
        df_prisma_90['CIF_UTE_REAL'] = df_prisma_90['CIF_UTE_REAL'].fillna(df_prisma_90['CIF_Norm'])
        df_prisma_90['Id UTE'] = df_prisma_90['Id UTE'].fillna('DESCONOCIDO')

        # Marcar si tiene match en PRISMA o no (para saber qué lógica usar en el solver)
        df_prisma_90['TIENE_MATCH_PRISMA'] = df_prisma_90['Num_Factura_Norm_P'].notna()

        # Importe: para las con match usamos PRISMA (con impuesto), para las sin match usamos COBRA directamente
        df_cobra_90_importes = df_cobra_90[['Num_Factura_Norm', col_importe]].copy()
        df_cobra_90_importes['IMPORTE_COBRA_DIRECTO'] = df_cobra_90_importes[col_importe].apply(convertir_importe_europeo)

        df_prisma_90 = df_prisma_90.merge(
            df_cobra_90_importes[['Num_Factura_Norm', 'IMPORTE_COBRA_DIRECTO']],
            on='Num_Factura_Norm',
            how='left'
        )

        # Sin match: usar importe de COBRA directamente (ya incluye impuesto)
        mask_sin_prisma = df_prisma_90['IMPORTE_CON_IMPUESTO'].isna() | (df_prisma_90['IMPORTE_CON_IMPUESTO'] == 0)
        df_prisma_90.loc[mask_sin_prisma, 'IMPORTE_CON_IMPUESTO'] = df_prisma_90.loc[mask_sin_prisma, 'IMPORTE_COBRA_DIRECTO']
        df_prisma_90['IMPORTE_CON_IMPUESTO'] = df_prisma_90['IMPORTE_CON_IMPUESTO'].fillna(0)
        df_prisma_90 = df_prisma_90.drop(columns=['IMPORTE_COBRA_DIRECTO'])

        # Fecha: para las sin match usar fecha de COBRA
        df_cobra_90_fechas = df_cobra_90[['Num_Factura_Norm', col_fecha_emision]].copy()
        df_cobra_90_fechas['FECHA_COBRA'] = pd.to_datetime(df_cobra_90_fechas[col_fecha_emision], dayfirst=True, errors='coerce')

        df_prisma_90 = df_prisma_90.merge(
            df_cobra_90_fechas[['Num_Factura_Norm', 'FECHA_COBRA']],
            on='Num_Factura_Norm',
            how='left'
        )

        if 'Fecha Emisión' not in df_prisma_90.columns:
            df_prisma_90['Fecha Emisión'] = pd.NaT

        mask_sin_fecha = df_prisma_90['Fecha Emisión'].isna()
        df_prisma_90.loc[mask_sin_fecha, 'Fecha Emisión'] = df_prisma_90.loc[mask_sin_fecha, 'FECHA_COBRA']
        df_prisma_90 = df_prisma_90.drop(columns=['FECHA_COBRA'])

        con_match = df_prisma_90['TIENE_MATCH_PRISMA'].sum()
        sin_match = (~df_prisma_90['TIENE_MATCH_PRISMA']).sum()
        st.write(f"- 90s de COBRA con match en PRISMA: **{con_match}**")
        st.write(f"- 90s de COBRA SIN match en PRISMA: **{sin_match}**")

        # DEBUG — dentro del if para que df_cobra_90 esté definido
        with st.expander("🔍 Info facturas 90 (base COBRA + enriquecido con PRISMA)"):
            facturas_90_positivas = (df_prisma_90['IMPORTE_CON_IMPUESTO'] > 0).sum()
            facturas_90_negativas = (df_prisma_90['IMPORTE_CON_IMPUESTO'] <= 0).sum()
            st.write(f"ℹ️ Total facturas 90 base (COBRA): {len(df_prisma_90)}")
            st.write(f"✅ Con match en PRISMA (socios TDE/TME disponibles): {con_match}")
            st.write(f"⚠️ Sin match en PRISMA (buscarán en COBRA TSOL): {sin_match}")
            st.write(f"✅ Con importe positivo: {facturas_90_positivas}")
            st.write(f"⛔ Con importe 0: {facturas_90_negativas}")
            st.dataframe(df_prisma_90[['Num_Factura_Norm', 'Id UTE', 'CIF_UTE_REAL', 'CIF_ORIGINAL', 'TIENE_MATCH_PRISMA', 'IMPORTE_CON_IMPUESTO']].head(20))

        # Informativo: solo en PRISMA, no en COBRA
        facturas_solo_prisma = df_prisma_90_base[
            ~df_prisma_90_base['Num_Factura_Norm'].isin(df_cobra_90['Num_Factura_Norm'])
        ]
        if len(facturas_solo_prisma) > 0:
            with st.expander(f"📋 Facturas 90 en PRISMA pero NO en COBRA (informativo: {len(facturas_solo_prisma)})"):
                st.info("Estas facturas existen en PRISMA pero no en COBRA — no se usarán en el cruce:")
                st.dataframe(
                    facturas_solo_prisma[['Num_Factura_Norm', 'CIF', 'CIF_Original_Norm', 'Id UTE', 'IMPORTE_CON_IMPUESTO']].head(50),
                    use_container_width=True
                )

        # Facturas negativas
        facturas_negativas = df_prisma_90[df_prisma_90['IMPORTE_CON_IMPUESTO'] < 0]
        if len(facturas_negativas) > 0:
            with st.expander(f"⚠️ Facturas 90 NEGATIVAS que se ignorarán ({len(facturas_negativas)} facturas)"):
                st.warning("Estas facturas negativas NO se considerarán en el cruce con pagos:")
                st.dataframe(
                    facturas_negativas[['Num_Factura_Norm', 'CIF_UTE_REAL', 'Id UTE', 'IMPORTE_CON_IMPUESTO']].sort_values('IMPORTE_CON_IMPUESTO'),
                    use_container_width=True
                )
                st.write(f"**Total importe negativo:** {facturas_negativas['IMPORTE_CON_IMPUESTO'].sum():,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

        st.session_state.df_prisma_90_preparado = df_prisma_90
        st.success(f"✅ Facturas 90 preparadas: {len(df_prisma_90)}")

    else:
        df_prisma_90 = st.session_state.df_prisma_90_preparado
        st.success(f"✅ Facturas 90 ya cargadas ({len(df_prisma_90)} facturas)")

    # -------------------------------
    # 3️⃣ FUNCIÓN OR-TOOLS CON SOCIOS
    # -------------------------------
    def cruzar_pagos_con_prisma_exacto(
        df_pagos, df_prisma_90, df_prisma_completo, df_cobra,
        col_num_factura_prisma, col_cif_cobra, col_sociedad_cobra,
        col_factura_cobra, col_importe_cobra, col_grupo_cobra,
        tolerancia=0.01
    ):
        """
        CASO 1: Factura 90 CON match en PRISMA
          → Socios son los de PRISMA (TDE/TME del mismo Id UTE)
          → Si la suma de socios PRISMA ≠ importe_90, buscar UNA factura TSOL en COBRA
            usando CIF_UTE_REAL (que es el CIF del socio/UTE ya normalizado)

        CASO 2: Factura 90 SIN match en PRISMA
          → Tomar CIF_ORIGINAL de la 90 en COBRA (CIF del cliente final)
          → Buscar en COBRA la columna col_grupo para ese CIF → obtener CIF grupal
          → Dentro del grupo, buscar filas con CIF que empiece por 'U' → UTEs
          → Entre esas UTEs, filtrar TSOL y buscar UNA factura que cuadre con importe_90
        """
        resultados = []

        # ── Preparar lookup CIF_ORIGINAL → CIF_GRUPO en COBRA ──────────────────
        # Construimos: cif_original → cif_grupo (usando todas las filas de COBRA)
        cif_a_grupo = {}
        if col_grupo_cobra:
            for _, row in df_cobra[[col_cif_cobra, col_grupo_cobra]].drop_duplicates().iterrows():
                cif_orig = str(row[col_cif_cobra]).strip()
                cif_grp  = str(row[col_grupo_cobra]).strip()
                if cif_orig and cif_grp and cif_grp.lower() not in ('nan', 'none', ''):
                    cif_a_grupo[cif_orig] = cif_grp

        # ── Preparar COBRA (no-TSS) indexada por CIF_GRUPO → CIF_UTE ───────────
        # Para CASO 2: dado un CIF_GRUPO buscamos facturas de UTEs del grupo
        # Un CIF de UTE empieza por 'U' (tras quitar el prefijo L-00)
        # Estructura: cobra_utes_por_grupo[cif_grupo][cif_ute_sin_prefijo] = DataFrame de facturas
        # Máximo UNA factura por sociedad (TSOL, TDE, TME...) dentro del mismo CIF UTE
        SOCIEDADES_TSS_EXCLUIR = {'TSS', 'TM01', 'T001'}
        cobra_utes_por_grupo = {}  # cif_grupo → {cif_ute → DataFrame de facturas no-TSS}
        if col_grupo_cobra and col_sociedad_cobra:
            df_no_tss = df_cobra[
                ~df_cobra[col_sociedad_cobra].astype(str).str.strip().str.upper().isin(SOCIEDADES_TSS_EXCLUIR)
            ].copy()
            df_no_tss['IMPORTE_COBRA']    = df_no_tss[col_importe_cobra].apply(convertir_importe_europeo)
            df_no_tss['NUM_FACTURA_COBRA'] = df_no_tss[col_factura_cobra].astype(str).str.strip()
            df_no_tss['SOCIEDAD_NORM']    = df_no_tss[col_sociedad_cobra].astype(str).str.strip().str.upper()
            df_no_tss['CIF_ORIGINAL']     = df_no_tss[col_cif_cobra].astype(str).str.strip()
            df_no_tss['CIF_SIN_PREFIJO']  = (
                df_no_tss['CIF_ORIGINAL']
                .str.replace(" ", "")
                .str.upper()
                .str.replace("L-00", "", regex=False)
            )
            df_no_tss['CIF_GRUPO_NORM']   = df_no_tss[col_grupo_cobra].astype(str).str.strip()
            # Incluir fecha de emisión si existe
            if col_fecha_emision in df_no_tss.columns:
                df_no_tss['FECHA_EMISION_COBRA'] = pd.to_datetime(df_no_tss[col_fecha_emision], dayfirst=True, errors='coerce')
            else:
                df_no_tss['FECHA_EMISION_COBRA'] = pd.NaT

            # Solo importe positivo
            df_no_tss = df_no_tss[df_no_tss['IMPORTE_COBRA'].notna() & (df_no_tss['IMPORTE_COBRA'] > 0)]

            for grp, grp_df in df_no_tss.groupby('CIF_GRUPO_NORM'):
                # Solo UTEs del grupo: CIF sin prefijo empieza por U
                utes_df = grp_df[grp_df['CIF_SIN_PREFIJO'].str.upper().str.startswith('U')]
                if utes_df.empty:
                    continue
                cobra_utes_por_grupo[grp] = {}
                for cif_ute, ute_df in utes_df.groupby('CIF_SIN_PREFIJO'):
                    cobra_utes_por_grupo[grp][cif_ute] = ute_df.copy()

        # ── Preparar COBRA no-TSS indexada por CIF normalizado (para CASO 1) ────
        SOCIEDADES_EXCLUIDAS = {'TSS', 'TM01', 'T001'}

        def normalizar_cif_cobra(valor):
            s = str(valor).replace(" ", "").strip().upper()
            s = s.replace("L-00", "")
            if s.startswith("C-"):
                s = s[2:]
            return s

        df_cobra_socios = df_cobra.copy()
        df_cobra_socios['CIF_UTE_NORM'] = df_cobra_socios[col_cif_cobra].apply(normalizar_cif_cobra)
        df_cobra_socios['SOCIEDAD_NORM'] = (
            df_cobra_socios[col_sociedad_cobra].astype(str).str.strip().str.upper()
            if col_sociedad_cobra else 'DESCONOCIDA'
        )
        df_cobra_socios['IMPORTE_COBRA'] = df_cobra_socios[col_importe_cobra].apply(convertir_importe_europeo)
        df_cobra_socios['NUM_FACTURA_COBRA'] = df_cobra_socios[col_factura_cobra].astype(str).str.strip()

        df_cobra_otros = df_cobra_socios[
            ~df_cobra_socios['SOCIEDAD_NORM'].isin(SOCIEDADES_EXCLUIDAS)
        ].copy()
        cobra_otros_por_cif = {cif: g for cif, g in df_cobra_otros.groupby('CIF_UTE_NORM')}

        # ── Socios PRISMA por Id UTE ─────────────────────────────────────────────
        df_socios = df_prisma_completo[
            (~df_prisma_completo[col_num_factura_prisma].astype(str).str.startswith("90")) &
            (df_prisma_completo['IMPORTE_CON_IMPUESTO'] > 0)
        ].copy()

        col_fecha_factura = None
        for posible_col in ['Fecha Emisión', 'Fecha Emision', 'FECHA_EMISION', 'Fecha']:
            if posible_col in df_prisma_completo.columns:
                col_fecha_factura = posible_col
                break

        socios_por_ute = {}
        for id_ute, grupo in df_socios.groupby('Id UTE'):
            cols_socios = [col_num_factura_prisma, 'IMPORTE_CON_IMPUESTO', 'CIF']
            if col_fecha_factura:
                cols_socios.append(col_fecha_factura)
            socios_por_ute[str(id_ute).strip()] = grupo[cols_socios].copy()

        # ── Agrupar facturas 90 por CIF_UTE_REAL ────────────────────────────────
        facturas_por_cif = {cif: g.copy() for cif, g in df_prisma_90.groupby('CIF_UTE_REAL')}

        # ── Bucle de pagos ───────────────────────────────────────────────────────
        for idx, pago in df_pagos.iterrows():
            try:
                cif_pago    = pago['CIF_UTE']
                importe_pago = pago['importe']
                fecha_pago  = pago['fec_operacion']

                if cif_pago not in facturas_por_cif:
                    resultados.append({
                        'CIF_UTE': cif_pago, 'fecha_pago': fecha_pago, 'importe_pago': importe_pago,
                        'facturas_90_asignadas': None, 'importe_facturas_90': 0.0,
                        'desglose_facturas_90': None, 'diferencia_pago_vs_90': importe_pago, 'advertencia': None
                    })
                    continue

                df_facturas = facturas_por_cif[cif_pago].copy()
                df_facturas = df_facturas[
                    (df_facturas['IMPORTE_CON_IMPUESTO'] > 0) &
                    (df_facturas['IMPORTE_CON_IMPUESTO'] <= importe_pago + tolerancia) &
                    (
                        df_facturas['Fecha Emisión'].isna() |
                        (df_facturas['Fecha Emisión'] <= fecha_pago)
                    )
                ].copy()

                if df_facturas.empty:
                    resultados.append({
                        'CIF_UTE': cif_pago, 'fecha_pago': fecha_pago, 'importe_pago': importe_pago,
                        'facturas_90_asignadas': 'SIN_FACTURAS_VALIDAS', 'importe_facturas_90': 0.0,
                        'desglose_facturas_90': None, 'diferencia_pago_vs_90': importe_pago, 'advertencia': None
                    })
                    continue

                df_facturas = df_facturas.sort_values(['Fecha Emisión', 'IMPORTE_CON_IMPUESTO'], ascending=[True, True])
                numeros_facturas  = df_facturas['Num_Factura_Norm'].tolist()
                importes_facturas = df_facturas['IMPORTE_CON_IMPUESTO'].tolist()
                ids_ute           = df_facturas['Id UTE'].tolist()
                fechas_90         = df_facturas['Fecha Emisión'].tolist()
                tiene_match       = df_facturas['TIENE_MATCH_PRISMA'].tolist()
                cifs_originales   = df_facturas['CIF_ORIGINAL'].tolist()  # CIF del cliente final (con L-00)

                importes_unicos       = len(set(importes_facturas))
                hay_facturas_duplicadas = importes_unicos < len(importes_facturas)

                # Solver OR-Tools para seleccionar qué facturas 90 cubren el pago
                model = cp_model.CpModel()
                n = len(importes_facturas)
                pagos_cent    = int(round(importe_pago * 100))
                facturas_cent = [int(round(f * 100)) for f in importes_facturas]
                tol_cent      = int(round(tolerancia * 100))
                x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
                model.Add(sum(x[i] * facturas_cent[i] for i in range(n)) >= pagos_cent - tol_cent)
                model.Add(sum(x[i] * facturas_cent[i] for i in range(n)) <= pagos_cent + tol_cent)
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 3
                solver.parameters.log_search_progress = False
                status = solver.Solve(model)

                if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                    seleccion = [i for i in range(n) if solver.Value(x[i]) == 1]
                    desglose_por_factura_90 = []
                    importe_facturas_90 = 0.0

                    for i in seleccion:
                        num_factura_90   = numeros_facturas[i]
                        importe_factura_90 = importes_facturas[i]
                        id_ute           = str(ids_ute[i]).strip()
                        fecha_factura_90 = fechas_90[i]
                        es_caso1         = tiene_match[i]       # True = CASO 1, False = CASO 2
                        cif_cliente_final = cifs_originales[i]  # CIF original de COBRA (con L-00)
                        importe_facturas_90 += importe_factura_90

                        socios_lista              = []
                        importe_socios_de_esta_90 = 0.0
                        socios_cobra              = []
                        importe_socios_cobra      = 0.0
                        estado_cobra              = None

                        # ── CASO 1: tiene match en PRISMA → socios TDE/TME ──────
                        if es_caso1:
                            if id_ute in socios_por_ute:
                                df_socios_ute = socios_por_ute[id_ute]
                                if col_fecha_factura and col_fecha_factura in df_socios_ute.columns:
                                    df_socios_ute = df_socios_ute[
                                        df_socios_ute[col_fecha_factura] <= fecha_factura_90
                                    ].copy()
                                for _, socio in df_socios_ute.iterrows():
                                    socios_lista.append({
                                        'num_factura': str(socio[col_num_factura_prisma]),
                                        'cif': str(socio['CIF']),
                                        'importe': socio['IMPORTE_CON_IMPUESTO'],
                                        'fuente': 'PRISMA'
                                    })
                                    importe_socios_de_esta_90 += socio['IMPORTE_CON_IMPUESTO']

                            diferencia_90_socios = round(importe_factura_90 - importe_socios_de_esta_90, 2)

                            # Si hay diferencia, buscar en COBRA (CASO 1 también puede necesitar TSOL)
                            if abs(diferencia_90_socios) > tolerancia:
                                cif_ute_real = df_facturas.iloc[i]['CIF_UTE_REAL'] if 'CIF_UTE_REAL' in df_facturas.columns else cif_pago
                                if cif_ute_real in cobra_otros_por_cif:
                                    df_cobra_cif = cobra_otros_por_cif[cif_ute_real].copy()
                                    importes_cobra = df_cobra_cif['IMPORTE_COBRA'].dropna().tolist()
                                    nums_cobra     = df_cobra_cif['NUM_FACTURA_COBRA'].tolist()
                                    socs_cobra     = df_cobra_cif['SOCIEDAD_NORM'].tolist()

                                    if importes_cobra:
                                        dif_cent   = int(round(diferencia_90_socios * 100))
                                        cobra_cent = [int(round(v * 100)) for v in importes_cobra]
                                        model_c = cp_model.CpModel()
                                        nc = len(cobra_cent)
                                        xc = [model_c.NewBoolVar(f"xc_{j}") for j in range(nc)]
                                        model_c.Add(sum(xc[j] * cobra_cent[j] for j in range(nc)) >= dif_cent - tol_cent)
                                        model_c.Add(sum(xc[j] * cobra_cent[j] for j in range(nc)) <= dif_cent + tol_cent)
                                        solver_c = cp_model.CpSolver()
                                        solver_c.parameters.max_time_in_seconds = 2
                                        solver_c.parameters.log_search_progress = False
                                        status_c = solver_c.Solve(model_c)

                                        if status_c in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                                            sel_c = [j for j in range(nc) if solver_c.Value(xc[j]) == 1]
                                            for j in sel_c:
                                                socios_cobra.append({
                                                    'num_factura': nums_cobra[j],
                                                    'cif': cif_ute_real,
                                                    'importe': importes_cobra[j],
                                                    'fuente': f'COBRA ({socs_cobra[j]})'
                                                })
                                                importe_socios_cobra += importes_cobra[j]
                                            estado_cobra = f"✅ Encontrado en COBRA ({', '.join(set(socs_cobra[j] for j in sel_c))})"
                                        else:
                                            estado_cobra = "⚠️ No se pudo cuadrar diferencia en COBRA (CASO 1)"
                                    else:
                                        estado_cobra = "⚠️ CIF encontrado en COBRA pero sin importes válidos"
                                else:
                                    estado_cobra = "❌ CIF no encontrado en COBRA para completar diferencia (CASO 1)"

                        # ── CASO 2: sin match en PRISMA → buscar socios en COBRA por grupo ──
                        else:
                            # cif_cliente_final = CIF de la factura 90 en COBRA (con L-00)
                            # Reglas:
                            #   - Máximo UNA factura por sociedad (TSOL/TDE/TME) dentro del MISMO CIF UTE
                            #   - Nunca mezclar facturas de diferentes CIFs UTE para cubrir una misma 90
                            #   - Fecha emisión ≤ fecha del pago
                            cif_grupo = cif_a_grupo.get(cif_cliente_final, None)
                            imp90_cent = int(round(importe_factura_90 * 100))

                            if cif_grupo and cif_grupo in cobra_utes_por_grupo:
                                utes_del_grupo = cobra_utes_por_grupo[cif_grupo]
                                mejor_resultado = None  # (socios_lista, importe_total, cif_ute)

                                for cif_ute, df_ute in utes_del_grupo.items():
                                    # Filtrar por fecha ≤ fecha del pago
                                    df_ute_fecha = df_ute[
                                        df_ute['FECHA_EMISION_COBRA'].isna() |
                                        (df_ute['FECHA_EMISION_COBRA'] <= fecha_pago)
                                    ].copy()

                                    if df_ute_fecha.empty:
                                        continue

                                    # Máximo UNA factura por sociedad dentro del mismo CIF UTE
                                    # Criterio: la más cercana en fecha POSTERIOR a la emisión de la 90
                                    # (fecha factura socio >= fecha emisión 90)
                                    # Si no hay fecha de la 90, usar cualquier factura anterior al pago
                                    filas_seleccionadas = []
                                    for soc, df_soc in df_ute_fecha.groupby('SOCIEDAD_NORM'):
                                        # Solo facturas con fecha >= fecha emisión de la 90
                                        if pd.notna(fecha_factura_90):
                                            df_soc_valida = df_soc[
                                                df_soc['FECHA_EMISION_COBRA'].isna() |
                                                (df_soc['FECHA_EMISION_COBRA'] >= fecha_factura_90)
                                            ].copy()
                                        else:
                                            df_soc_valida = df_soc.copy()

                                        if df_soc_valida.empty:
                                            continue

                                        # La más cercana a la fecha de emisión de la 90 (más antigua válida)
                                        if df_soc_valida['FECHA_EMISION_COBRA'].notna().any():
                                            df_soc_valida = df_soc_valida.sort_values('FECHA_EMISION_COBRA', ascending=True)
                                        filas_seleccionadas.append(df_soc_valida.iloc[[0]])

                                    if not filas_seleccionadas:
                                        continue
                                    df_una_por_soc = pd.concat(filas_seleccionadas, ignore_index=True)

                                    # Filtrar las que individualmente no superan el importe de la 90
                                    df_una_por_soc = df_una_por_soc[
                                        df_una_por_soc['IMPORTE_COBRA'] <= importe_factura_90 + tolerancia
                                    ].copy()

                                    if df_una_por_soc.empty:
                                        continue

                                    importes_ute  = df_una_por_soc['IMPORTE_COBRA'].tolist()
                                    nums_ute      = df_una_por_soc['NUM_FACTURA_COBRA'].tolist()
                                    socs_ute      = df_una_por_soc['SOCIEDAD_NORM'].tolist()
                                    nte           = len(importes_ute)
                                    ute_cent      = [int(round(v * 100)) for v in importes_ute]

                                    model_t = cp_model.CpModel()
                                    xt = [model_t.NewBoolVar(f"xt_{cif_ute}_{j}") for j in range(nte)]
                                    # Suma debe cuadrar con importe de la 90
                                    model_t.Add(sum(xt[j] * ute_cent[j] for j in range(nte)) >= imp90_cent - tol_cent)
                                    model_t.Add(sum(xt[j] * ute_cent[j] for j in range(nte)) <= imp90_cent + tol_cent)
                                    solver_t = cp_model.CpSolver()
                                    solver_t.parameters.max_time_in_seconds = 2
                                    solver_t.parameters.log_search_progress = False
                                    status_t = solver_t.Solve(model_t)

                                    if status_t in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                                        sel_t = [j for j in range(nte) if solver_t.Value(xt[j]) == 1]
                                        candidatos = []
                                        importe_candidato = 0.0
                                        for j in sel_t:
                                            candidatos.append({
                                                'num_factura': nums_ute[j],
                                                'cif': cif_ute,
                                                'importe': importes_ute[j],
                                                'fuente': f'COBRA ({socs_ute[j]})'
                                            })
                                            importe_candidato += importes_ute[j]
                                        # Guardar si es la primera solución válida
                                        if mejor_resultado is None:
                                            mejor_resultado = (candidatos, importe_candidato, cif_ute)
                                        # Preferir solución con diferencia más pequeña
                                        elif abs(importe_candidato - importe_factura_90) < abs(mejor_resultado[1] - importe_factura_90):
                                            mejor_resultado = (candidatos, importe_candidato, cif_ute)

                                if mejor_resultado is not None:
                                    socios_cobra        = mejor_resultado[0]
                                    importe_socios_cobra = mejor_resultado[1]
                                    socs_encontradas    = ', '.join(set(s['fuente'] for s in socios_cobra))
                                    estado_cobra = f"✅ Socios encontrados en CIF UTE {mejor_resultado[2]} (grupo {cif_grupo}): {socs_encontradas}"
                                else:
                                    estado_cobra = f"⚠️ Ningún CIF UTE del grupo {cif_grupo} cuadra con {importe_factura_90:.2f}€"
                            elif cif_grupo:
                                estado_cobra = f"⚠️ Grupo {cif_grupo} no tiene CIFs UTE con facturas válidas"
                            else:
                                estado_cobra = f"❌ No se encontró CIF grupal para {cif_cliente_final}"

                        # Combinar socios PRISMA + socios COBRA
                        todos_socios              = socios_lista + socios_cobra
                        importe_total_socios_final = importe_socios_de_esta_90 + importe_socios_cobra
                        diferencia_final           = round(importe_factura_90 - importe_total_socios_final, 2)

                        desglose_por_factura_90.append({
                            'factura_90':            num_factura_90,
                            'importe_90':            importe_factura_90,
                            'caso':                  'CASO 1 (PRISMA)' if es_caso1 else 'CASO 2 (sin PRISMA)',
                            'cif_cliente_final':     cif_cliente_final,
                            'socios':                todos_socios,
                            'importe_socios':        importe_total_socios_final,
                            'diferencia_90_socios':  diferencia_final,
                            'socios_prisma':         socios_lista,
                            'socios_cobra':          socios_cobra,
                            'importe_socios_prisma': importe_socios_de_esta_90,
                            'importe_socios_cobra':  importe_socios_cobra,
                            'estado_cobra':          estado_cobra
                        })

                    facturas_90_str = ', '.join([d['factura_90'] for d in desglose_por_factura_90])

                else:
                    facturas_90_str         = None
                    importe_facturas_90     = 0.0
                    desglose_por_factura_90 = None
                    hay_facturas_duplicadas = False

                diferencia_pago_vs_90 = importe_pago - importe_facturas_90
                advertencia = None
                if hay_facturas_duplicadas and facturas_90_str is not None:
                    advertencia = "⚠️ ATENCIÓN: Había múltiples facturas 90 con importes similares. Se priorizaron las más antiguas."

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

    st.write(f"⚠️ Filas en df_prisma_90 ANTES del solver: {len(df_prisma_90)}")
    st.write(f"⚠️ Facturas únicas: {df_prisma_90['Num_Factura_Norm'].nunique()}")
    st.write(f"⚠️ CIFs únicos: {df_prisma_90['CIF_UTE_REAL'].nunique()}")

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
                df,                      # DataFrame COBRA completo
                col_num_factura_prisma,
                col_cif,                 # Columna CIF de COBRA
                col_sociedad,            # Columna Sociedad de COBRA
                col_factura,             # Columna Factura de COBRA
                col_importe,             # Columna Importe de COBRA
                col_grupo,               # Columna CIF Grupo de COBRA  ← NUEVO
                0.01
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
