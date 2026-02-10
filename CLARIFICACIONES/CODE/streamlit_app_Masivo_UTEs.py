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




# --------- 1) Subida y normalización de PRISMA ---------
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






        # --------- 2) subida y normalizacion de COBRA ---------
archivo = st.file_uploader("Sube el archivo Excel DetalleDocumentos de Cobra", type=["xlsx", "xls"])
if archivo:
    # --- Lectura flexible para detectar cabecera ---
    try:
        df_raw = pd.read_excel(archivo, engine="openpyxl", header=None)
    except Exception:
        df_raw = pd.read_excel(archivo, header=None)

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
    try:
        df = pd.read_excel(archivo, engine="openpyxl", header=header_row)
    except Exception:
        df = pd.read_excel(archivo, header=header_row)

    with st.expander("🔎 Ver columnas detectadas en el Excel"):
        st.write(list(df.columns))

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
    if not col_nombre_grupo:    faltan.append("Nombre grupo")
    if faltan:
        st.error("❌ No se pudieron localizar estas columnas: " + ", ".join(faltan))
        st.stop()

    # --- Normalizar ---
    df[col_fecha_emision] = pd.to_datetime(df[col_fecha_emision], dayfirst=True, errors='coerce')
    df[col_factura] = df[col_factura].astype(str)
    df[col_cif] = df[col_cif].astype(str)
    df['IMPORTE_CORRECTO'] = df[col_importe].apply(convertir_importe_europeo)
    df['IMPORTE_CENT'] = (df['IMPORTE_CORRECTO'] * 100).round().astype("Int64")

    #Resumen del archivo
    total = df['IMPORTE_CORRECTO'].sum(skipna=True)
    minimo = df['IMPORTE_CORRECTO'].min(skipna=True)
    maximo = df['IMPORTE_CORRECTO'].max(skipna=True)
    
    st.write("**📊 Resumen del archivo:**")
    st.write(f"- Número total de facturas: {len(df)}")
    st.write(f"- Suma total importes: {total:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe mínimo: {minimo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe máximo: {maximo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

    # --- Detectar UTES ---
    df['ES_UTE'] = df[col_cif].astype(str).str.replace(" ", "").str.contains(r"L-00U")

    # --- Selección de modo de búsqueda ---
    modo_busqueda = st.radio(
        "🔹 Selecciona el modo de búsqueda:",
        ("Por factura TSS (90)", "Por cliente/grupo")
    )

    # Inicializar variables para que existan en todo el scope
    grupo_seleccionado = None
    factura_final = None
    df_filtrado = pd.DataFrame()
    df_tss = pd.DataFrame()
    df_internas = pd.DataFrame()
    df_tss_selec = pd.DataFrame()
    df_resultado_final = pd.DataFrame()
    df_resultado = pd.DataFrame()   





# --- 3) Subida de archivo de pagos (Cobros UTE) ---
    cobros_file = st.file_uploader(
        "Sube el Excel de pagos de UTE ej. Informe_Cruce_Movimientos 19052025 a 19082025",
        type=['xlsm', 'xlsx', 'csv'],
        key="cobros"
    )

    df_cobros = pd.DataFrame()
    if cobros_file:
        try:
            if cobros_file.name.endswith(('.xlsm', '.xlsx')):
                # Guardamos en BytesIO para poder leer varias veces
                data = BytesIO(cobros_file.read())

                # 1) Detectar hojas
                xls = pd.ExcelFile(data, engine="openpyxl")
                

                # 2) Seleccionar la hoja
                sheet = "Cruce_Movs" if "Cruce_Movs" in xls.sheet_names else xls.sheet_names[0]

                # resetear puntero y leer la hoja
                data.seek(0)
                df_cobros = pd.read_excel(data, sheet_name=sheet, engine="openpyxl")

            else:  # CSV
                df_cobros = pd.read_csv(cobros_file, sep=None, engine="python")

        
        except Exception as e:
            st.error(f"Error al leer el archivo de pagos: {e}")
            df_cobros = pd.DataFrame()


    # Si no hay resultado interno, paramos aquí (nada que asignar)
    if df_cobros.empty:
            st.info("ℹ️ Sube un archivo de pagos para comenzar el cruce.")
            st.stop()

    else:
        # Normalizamos columnas de df_cobros para poder mapear
        if not df_cobros.empty:
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

        
            # Mapeo seguro de columnas que usamos
            col_map = {
                'fec_operacion': ['fec_operacion', 'fecha_operacion', 'fec_oper'],
                'importe': ['importe', 'imp', 'monto', 'amount', 'valor'],
                'posible_factura': ['posible_factura', 'factura', 'posiblefactura'],
                'norma_43': ['norma_43', 'norma43']
            }
            for target, possibles in col_map.items():
                for p in possibles:
                    if p in df_cobros.columns:
                        df_cobros.rename(columns={p: target}, inplace=True)
                        break

            # aseguramos tipos
            if 'fec_operacion' in df_cobros.columns:
                df_cobros['fec_operacion'] = pd.to_datetime(df_cobros['fec_operacion'], errors='coerce')
            if 'importe' in df_cobros.columns:
                df_cobros['importe'] = pd.to_numeric(df_cobros['importe'], errors='coerce')
            # columnas textuales
            if 'posible_factura' in df_cobros.columns:
                df_cobros['posible_factura'] = df_cobros['posible_factura'].astype(str).str.strip()
            if 'norma_43' in df_cobros.columns:
                df_cobros['norma_43'] = df_cobros['norma_43'].astype(str).str.strip()


            if not df_cobros.empty:
                st.subheader("🔍 Debug rápido de Cruce_Movs")

                # Estadísticas básicas
                num_filas = len(df_cobros)
                total_importes = df_cobros['importe'].sum(skipna=True)
                min_importe = df_cobros['importe'].min(skipna=True)
                max_importe = df_cobros['importe'].max(skipna=True)
                pagos_con_factura = df_cobros['posible_factura'].notna().sum() if 'posible_factura' in df_cobros.columns else 0

                st.write(f"- Número de filas: {num_filas}")
                st.write(f"- Suma total de importes: {total_importes:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                st.write(f"- Importe mínimo: {min_importe:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                st.write(f"- Importe máximo: {max_importe:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
                st.write(f"- Pagos con posible factura: {pagos_con_factura}")

                # Primeras filas para inspección
                st.dataframe(df_cobros.head(10), use_container_width=True)

    