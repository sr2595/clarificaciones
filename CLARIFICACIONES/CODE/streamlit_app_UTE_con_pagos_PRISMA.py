import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO
from datetime import datetime
import unicodedata, re
import io
import os

# --------- Función solver para internas ------------
def cuadrar_internas(externa, df_internas, tol=0):
    if externa is None or df_internas.empty:
        return pd.DataFrame()

    objetivo = int(externa['IMPORTE_CENT'])
    fecha_ref = externa[col_fecha_emision]

    data = list(zip(
        df_internas.index.tolist(),
        df_internas['IMPORTE_CENT'].astype(int).tolist(),
        (df_internas[col_fecha_emision] - fecha_ref).dt.days.fillna(0).astype(int).tolist(),
        df_internas[col_sociedad].tolist()
    ))

    n = len(data)
    if n == 0:
        return pd.DataFrame()

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"sel_{i}") for i in range(n)]

    # Exacto o con tolerancia
    if tol == 0:
        model.Add(sum(x[i] * data[i][1] for i in range(n)) == objetivo)
    else:
        model.Add(sum(x[i] * data[i][1] for i in range(n)) >= objetivo - tol)
        model.Add(sum(x[i] * data[i][1] for i in range(n)) <= objetivo + tol)

    # Restricción: no repetir sociedad
    sociedades = set(d[3] for d in data)
    for s in sociedades:
        indices = [i for i, d in enumerate(data) if d[3] == s]
        if indices:
            model.Add(sum(x[i] for i in indices) <= 1)

    # Minimizar desviación de fechas (opcional)
    costs = [abs(d[2]) for d in data]
    model.Minimize(sum(x) + sum(x[i] * costs[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        seleccionadas = [data[i][0] for i in range(n) if solver.Value(x[i]) == 1]
        return df_internas.loc[seleccionadas]
    else:
        return pd.DataFrame()

st.write("DEBUG archivo en ejecución:", os.path.abspath(__file__))

st.set_page_config(page_title="Clarificador UTE con pagos", page_icon="📄", layout="wide")
st.title("📄 Clarificador UTE con pagos")

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

# --------- Hook PRISMA ---------
        def hook_prisma(factura_final, df_prisma, col_num_factura_prisma, col_cif_prisma, col_importe_prisma, col_id_ute_prisma, col_tipo_impuesto=col_tipo_imp_prisma):
            prisma_cubierto = False
            pendiente_prisma = None

            # Función auxiliar para aplicar impuesto según la columna tipo_impuesto
            def aplicar_impuesto_prisma(importe, tipo_impuesto):
                factores = {
                    "IGIC - 7": 1.07,
                    "IPSIC - 10": 1.10,
                    "IPSIM - 8": 1.08,
                    "IVA - 0": 1.00,
                    "IVA - 21": 1.21,
                    "EXENTO": 1.0,
                    "IVA - EXENTO": 1.0,
                }
                return float(importe * factores.get(str(tipo_impuesto).strip().upper(), 1.0))

            try:
                factura_90_val = str(factura_final[col_factura]).strip()
            except Exception:
                factura_90_val = None

            if factura_90_val:
                fila_90_prisma = df_prisma[df_prisma[col_num_factura_prisma].astype(str).str.strip() == factura_90_val]

                if fila_90_prisma.empty:
                    st.warning(f"⚠️ La factura {factura_90_val} NO se encuentra en PRISMA. Se continuará usando solo COBRA.")
                    prisma_cubierto = False
                else:
                    fila_90_prisma = fila_90_prisma.iloc[0]
                    fecha_90_prisma = pd.to_datetime(
                        fila_90_prisma.get(col_fecha_emision, None),
                        errors="coerce")                                        

                    id_ute_90 = str(fila_90_prisma[col_id_ute_prisma]).strip()
                    st.success(f"✅ Factura 90 encontrada en PRISMA. id UTE = {id_ute_90}")

                    df_parejas = df_prisma[df_prisma[col_id_ute_prisma].astype(str).str.strip() == id_ute_90].copy()
                    df_socios_prisma = df_parejas[df_parejas[col_num_factura_prisma].astype(str).str.strip() != factura_90_val].copy()

                    # Aplicar impuesto a los importes
                    df_socios_prisma['importe_con_impuesto'] = df_socios_prisma.apply(
                        lambda row: aplicar_impuesto_prisma(row['IMPORTE_CORRECTO'], row.get(col_tipo_impuesto, 'EXENTO')), axis=1
                    )
                    fila_90_prisma['importe_con_impuesto'] = aplicar_impuesto_prisma(fila_90_prisma['IMPORTE_CORRECTO'], fila_90_prisma.get(col_tipo_impuesto, 'EXENTO'))

                    st.write("Columnas en df_prisma / df_parejas:")
                    st.write(list(df_prisma.columns))
                    st.write(list(df_parejas.columns))

                    st.subheader("📂 PRISMA: filas relacionadas con id UTE")
                    st.write(f"Filas totales con id UTE = {len(df_parejas)} (excluyendo la 90 -> {len(df_socios_prisma)})")
                    st.dataframe(df_parejas[[ col_cif_prisma, col_fecha_emision, col_num_factura_prisma, col_importe_prisma, col_tipo_impuesto]].head(30), use_container_width=True)

                    importe_90_prisma = fila_90_prisma.get('importe_con_impuesto', 0.0)
                    importe_socios_prisma = float(df_socios_prisma['importe_con_impuesto'].sum()) if not df_socios_prisma.empty else 0.0
                    diferencia = (importe_90_prisma or 0.0) - importe_socios_prisma

                    st.info(f"💶 PRISMA → importe 90 (con impuesto): {importe_90_prisma:,.2f} €")
                    st.info(f"💶 PRISMA → suma socios (TDE/TME) con impuesto: {importe_socios_prisma:,.2f} €")
                    st.info(f"🔢 PRISMA → diferencia (90 - socios): {diferencia:,.2f} €")

                    tol_euros = 0.01
                    if abs(diferencia) <= tol_euros:
                        prisma_cubierto = True
                        st.success("🎉 La UTE queda cuadrada con PRISMA (no hace falta buscar en COBRA).")
                        st.session_state["resultado_prisma_directo"] = {
                            "id_ute": id_ute_90,
                            "factura_90": factura_90_val,
                            "socios_df": df_socios_prisma,
                            "fila_90": fila_90_prisma
                        }
                    else:
                        prisma_cubierto = False
                        resto_cent = int(round(diferencia * 100))
                        st.warning(f"⚠️ PRISMA no cubre totalmente la 90 — resta {diferencia:,.2f} € ({resto_cent} cént.) que habrá que cuadrar en COBRA")
                        pendiente_prisma = {
                            "id_ute": id_ute_90,
                            "factura_90": factura_90_val,
                            "fecha_90_prisma": fecha_90_prisma,
                            "importe_90_prisma": importe_90_prisma,
                            "importe_socios_prisma": importe_socios_prisma,
                            "resto_euros": diferencia,
                            "resto_cent": resto_cent,
                            "df_socios_prisma": df_socios_prisma
                        }
                        st.session_state["pendiente_prisma"] = pendiente_prisma


            return prisma_cubierto, pendiente_prisma


# --------- subida y normalizacion de COBRA ---------
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

    if modo_busqueda == "Por factura TSS (90)":
        # --- Input alternativo: buscar directamente por factura TSS (90) ---
        factura_input = st.text_input("🔎 Buscar por nº de factura TSS (90)").strip()

        if factura_input:
            # Buscar esa factura en TSS
            df_tss_all = df[df[col_sociedad].astype(str).str.upper().str.strip() == "TSS"].copy()
            factura_input_norm = str(factura_input).strip()
            mask_fact = df_tss_all[col_factura].astype(str).str.strip() == factura_input_norm

            if mask_fact.any():
                # Seleccionamos la factura encontrada
                factura_final = df_tss_all.loc[mask_fact].iloc[0]
                grupo_seleccionado = str(factura_final[col_grupo]).replace(" ", "")
                cliente_final_nombre = factura_final[col_nombre_cliente] if col_nombre_cliente else ""
                
                st.success(
                    f"Factura encontrada: **{factura_final[col_factura]}** "
                    f"({factura_final['IMPORTE_CORRECTO']:,.2f} €) | Fecha emisión: {factura_final[col_fecha_emision].date()} | Grupo: {grupo_seleccionado} | Cliente: {cliente_final_nombre}"
                )

                # Filtramos todo el grupo asociado a esa factura
                df_filtrado = df[df[col_grupo].astype(str).str.replace(" ", "") == grupo_seleccionado].copy()

                # Filtramos facturas TSS de ese grupo
                df_tss = df_filtrado[df_filtrado[col_sociedad].astype(str).str.upper().str.strip() == 'TSS']

                # Seleccionamos como factura final la que buscó el usuario
                df_factura_final = df_tss[df_tss[col_factura].astype(str).str.strip() == factura_input_norm]
                if not df_factura_final.empty:
                    factura_final = df_factura_final.iloc[0]
                else:
                    st.error(f"❌ La factura {factura_input_norm} no se encuentra tras filtrar el grupo.")
                    factura_final = None

                    
                 # 🔹 Llamada al hook PRISMA
                if factura_final is not None and not df_prisma.empty:
                    prisma_cubierto, pendiente_prisma = hook_prisma(
                        factura_final,
                        df_prisma,
                        col_num_factura_prisma,
                        col_cif_prisma,
                        col_importe_prisma,
                        col_id_ute_prisma
                    )
                    if pendiente_prisma is not None:
                        # 🔹 Limpiar CIF en df y extraer solo la parte real del CIF (alfanumérica)
        
                        df['CIF_LIMPIO'] = (
                            df[col_cif].astype(str)
                            .str.replace(r"[^A-Za-z0-9]", "", regex=True)  # deja solo letras y números
                            .str.upper()
                        )
                        # 🔹 Quitar cualquier letra inicial seguida de 0
                        df['CIF_LIMPIO'] = df['CIF_LIMPIO'].str.replace(r'^[A-Z]00', '', regex=True)

                        # 🔹 Obtener todos los CIFs de los socios de la UTE que generan pendiente
                        socios_prisma = pendiente_prisma['df_socios_prisma'][col_cif_prisma].tolist()
                        socios_prisma_limpios = socios_prisma_limpios = [re.sub(r"[^A-Za-z0-9]", "", str(s)).upper() for s in socios_prisma]
                        socios_prisma_limpios = [re.sub(r'^[A-Z]00', '', s) for s in socios_prisma_limpios]

                        # 🔹 Rellenar df_internas automáticamente con todas las internas de esos socios
                        df_internas = df[df['CIF_LIMPIO'].isin(socios_prisma_limpios)].copy()

                        # 🔹 Filtrar solo sociedades internas relevantes
                        df_internas = df_internas[df_internas[col_sociedad].astype(str).str.upper().isin(['TSOL', 'TDE', 'TME'])]

                        # 🔹 DEBUG: mostrar incluso si está vacío
                        st.subheader("🧪 DEBUG PRISMA → COBRA (TSOL) — df_internas rellenado automáticamente")
                        st.write(f"CIF UTE limpio: {socios_prisma_limpios}")
                        st.write(f"Filas encontradas: {len(df_internas)}")
                        st.dataframe(df_internas[['CIF_LIMPIO', col_factura, col_sociedad, "IMPORTE_CORRECTO", col_fecha_emision]], use_container_width=True)


                        # 🔹 7️⃣ Opcional: mostrar todas las sociedades y CIFs presentes para verificar coincidencias
                        st.write("CIFs en df:", df['CIF_LIMPIO'].astype(str).unique())
                        st.write("Sociedades disponibles en df:", df[col_sociedad].astype(str).unique())

                        # 🔹 Priorizar internas por cercanía a la fecha 90 PRISMA
                        fecha_ref = pendiente_prisma.get("fecha_90_prisma")

                        if fecha_ref is not None and col_fecha_emision in df_internas.columns:
                            df_internas = df_internas.copy()
                            df_internas[col_fecha_emision] = pd.to_datetime(
                                df_internas[col_fecha_emision],
                                errors="coerce"
                            )

                            df_internas["DIST_FECHA_90"] = (
                                df_internas[col_fecha_emision] - fecha_ref
                            ).abs()

                            df_internas = df_internas.sort_values(
                                by=["DIST_FECHA_90", col_fecha_emision]
                            )
                                               
                        # Ejecutar solver COBRA con el restante PRISMA
                        df_resultado_restante = cuadrar_internas(
                        pd.Series({
                            'IMPORTE_CENT': pendiente_prisma["resto_cent"],
                            col_fecha_emision: pendiente_prisma.get("fecha_90_prisma")
                        }),
                        df_internas )
                                  

                        if not df_resultado_restante.empty:
                            st.success(f"✅ Se cuadró el restante de PRISMA ({pendiente_prisma['resto_euros']:,.2f} €) con COBRA")
                            st.dataframe(
                                df_resultado_restante[[ col_cif, col_nombre_cliente, col_factura,'IMPORTE_CORRECTO', col_fecha_emision]],
                                use_container_width=True
                            )
                        else:
                            st.warning("⚠️ No se encontró combinación de facturas internas que cuadre con el restante de PRISMA")

                    if prisma_cubierto:
                        res = st.session_state.get("resultado_prisma_directo", {})
                        if res:
                            st.subheader("✅ Resultado final (PRISMA)")
                            st.write(f"ID UTE: {res['id_ute']}, Factura 90: {res['factura_90']}")
                            st.dataframe(
                                res['socios_df'][
                                    [col_num_factura_prisma, col_cif_prisma, col_importe_prisma, 'IMPORTE_CORRECTO']
                                ],
                                use_container_width=True
                            )

                    else:
                        # ⚠️ PRISMA no cubre completamente
                        if pendiente_prisma is not None and not df_internas.empty:

                            # ----------------------------------
                            # 1️⃣ Construir externa pendiente
                            # ----------------------------------
                            externa_pendiente = pd.Series({
                                'IMPORTE_CENT': int(pendiente_prisma["resto_cent"]),
                                col_fecha_emision: (
                                    factura_final[col_fecha_emision]
                                    if col_fecha_emision in factura_final
                                    else factura_final.get(col_fecha_emision, pd.NaT)
                                )
                            })

                            # ----------------------------------
                            # 2️⃣ Filtrar internas por CIF UTE (usar columna limpia)
                            # ----------------------------------
                            df_internas_filtrado = df_internas[
                                df_internas['CIF_LIMPIO'].isin(socios_prisma_limpios)
                            ].copy()

                                             
                else:
                    st.error(f"❌ No se encontró la factura TSS nº {factura_input_norm}")
                    st.stop()
  





    elif modo_busqueda == "Por cliente/grupo":
     
            # --- Opciones de grupos ---
            df[col_grupo] = df[col_grupo].astype(str).str.replace(" ", "")
            df[col_nombre_grupo] = df[col_nombre_grupo].fillna("").str.strip()
            df_grupos_unicos = (
                df[[col_grupo, col_nombre_grupo]]
                .drop_duplicates()
                .sort_values([col_nombre_grupo, col_grupo])
            )
            opciones_grupos = [
                f"{row[col_grupo]} - {row[col_nombre_grupo]}" if row[col_nombre_grupo] else f"{row[col_grupo]}"
                for _, row in df_grupos_unicos.iterrows()
            ]
            grupo_seleccionado_display = st.selectbox("Selecciona CIF grupal", opciones_grupos)
            grupo_seleccionado = grupo_seleccionado_display.split(" - ")[0]
            st.write("Grupo seleccionado (CIF):", grupo_seleccionado)

            # --- Filtrar TSS del grupo ---
            df_filtrado = df[df[col_grupo] == grupo_seleccionado].copy()
            df_tss = df_filtrado[df_filtrado[col_sociedad].astype(str).str.upper().str.strip() == "TSS"]

            
            # --- Input opcional: importe de pago para solver de TSS ---
            importe_pago_str = st.text_input("💶 Introduce importe de pago (opcional, formato europeo: 96.893,65)")
            tolerancia_str = st.text_input("🎯 Tolerancia en céntimos (opcional, 0 = exacto, ej: 100 = ±1€), si no indicas nada no aplicara tolerancia y buscara el importe exacto", "0")


            def parse_importe_europeo(texto):
                if not texto:
                    return None
                texto = str(texto).replace(" ", "").replace(".", "").replace(",", ".")
                try:
                    return float(texto)
                except:
                    return None

            importe_pago = parse_importe_europeo(importe_pago_str)
            try:
                tolerancia_cent = int(tolerancia_str)
                if tolerancia_cent < 0:
                    tolerancia_cent = 0
            except:
                tolerancia_cent = 0

            if importe_pago is not None and importe_pago > 0 and not df_tss.empty:

                def solver_tss_pago(df_tss, importe_pago, tol=0):
                    from ortools.sat.python import cp_model

                    if df_tss.empty or importe_pago is None:
                        return pd.DataFrame()

                    df_tss = df_tss[df_tss['IMPORTE_CORRECTO'] > 0].copy()
                    if df_tss.empty:
                        return pd.DataFrame()

                    # Deduplicar por sociedad+factura
                    if col_sociedad in df_tss.columns and col_factura in df_tss.columns:
                        df_tss['_clave_unica'] = df_tss[col_sociedad].astype(str) + "_" + df_tss[col_factura].astype(str)
                        df_tss = df_tss.drop_duplicates(subset=['_clave_unica'])

                    # Control global de facturas usadas
                    socios_facturas_usadas = set()
                    seleccion_total = []

                    # Resolver cliente por cliente
                    for cif, df_cliente in df_tss.groupby(col_cif):
                        df_cliente = df_cliente.copy()
                        df_cliente['IMPORTE_CENT'] = (df_cliente['IMPORTE_CORRECTO'] * 100).round().astype("Int64")
                        objetivo = int(importe_pago * 100)

                        data = list(zip(df_cliente.index.tolist(), df_cliente['IMPORTE_CENT'].tolist()))
                        n = len(data)

                        model = cp_model.CpModel()
                        x = [model.NewBoolVar(f"sel_{i}") for i in range(n)]

                        # Suma ≈ objetivo
                       # 🎯 Exacto o con tolerancia según input
                        if tol == 0:
                            model.Add(sum(x[i] * data[i][1] for i in range(n)) == objetivo)
                        else:
                            model.Add(sum(x[i] * data[i][1] for i in range(n)) >= objetivo - tol)
                            model.Add(sum(x[i] * data[i][1] for i in range(n)) <= objetivo + tol)


                        # Restricción: cada factura (sociedad+numero) solo una vez en TODO el flujo
                        for i, idx in enumerate(df_cliente.index):
                            clave = (df_cliente.at[idx, col_sociedad], df_cliente.at[idx, col_factura])
                            if clave in socios_facturas_usadas:
                                model.Add(x[i] == 0)  # ❌ No se puede seleccionar si ya fue usada

                        # Restricción: no repetir factura dentro del mismo cliente
                        for (soc, fac), g in df_cliente.groupby([col_sociedad, col_factura]):
                            idxs = [i for i, idx in enumerate(df_cliente.index) if idx in g.index]
                            if len(idxs) > 1:
                                model.Add(sum(x[i] for i in idxs) <= 1)

                        solver = cp_model.CpSolver()
                        solver.parameters.max_time_in_seconds = 10
                        status = solver.Solve(model)

                        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                            seleccionadas = [data[i][0] for i in range(n) if solver.Value(x[i]) == 1]
                            df_selec_cliente = df_cliente.loc[seleccionadas]
                            seleccion_total.append(df_selec_cliente)

                            # Marcar facturas globalmente usadas
                            socios_facturas_usadas.update(
                                df_selec_cliente[[col_sociedad, col_factura]].itertuples(index=False, name=None)
                            )
                    if seleccion_total:
                        df_out = pd.concat(seleccion_total)
                        df_out = df_out.drop_duplicates(subset=[col_sociedad, col_factura])
                        return df_out


                    return pd.DataFrame()


                # --- 2) Llamada al solver si se introduce importe de pago ---
                solver_used = False
                df_tss_selec = solver_tss_pago(df_tss.copy(), importe_pago, tol=tolerancia_cent)

                if not df_tss_selec.empty:
                    df_resultado = df_tss_selec.copy()
                    # Deduplicar por seguridad antes de agregar info de pago
                    df_resultado = df_resultado.drop_duplicates(subset=[col_sociedad, col_factura])
                    solver_used = True
                    st.success(f"✅ Se encontró combinación de {len(df_tss_selec)} facturas TSS que suman {df_tss_selec['IMPORTE_CORRECTO'].sum():,.2f} €")
                    st.dataframe(df_tss_selec[[col_cif, col_nombre_cliente, col_factura, col_fecha_emision, 'IMPORTE_CORRECTO']], use_container_width=True)

                # --- Si el solver se usó, solo entonces creamos la factura agrupada como fallback
                if solver_used:
                    # Si NO hay facturas internas seleccionadas por el usuario
                    if df_internas.empty:
                        # Tomamos las facturas seleccionadas por el solver
                        df_resultado = df_tss_selec.copy()

                        # Deduplicar por socio + factura (por seguridad)
                        if not df_resultado.empty and col_factura in df_resultado and col_sociedad in df_resultado:
                            df_resultado = df_resultado.drop_duplicates(subset=[col_factura, col_sociedad])

                        # Crear factura final agrupada
                        total_importe = float(df_resultado["IMPORTE_CORRECTO"].sum())
                        fecha_min = df_resultado[col_fecha_emision].min()

                        factura_final = pd.Series({
                            col_cif: "AGRUPADO",
                            col_nombre_cliente: "Facturas TSS agrupadas",
                            col_factura: "AGRUPADO",
                            col_fecha_emision: fecha_min,
                            "IMPORTE_CORRECTO": total_importe,
                            "IMPORTE_CENT": int(round(total_importe * 100))
                        })

                    else:
                        # Si hay internas, dejamos que el flujo normal actúe
                        df_resultado = pd.DataFrame()



            else:
                # Flujo normal: selección de cliente final y filtrado de TSS
                # --- Opciones de clientes finales del grupo ---
                df[col_cif] = df[col_cif].astype(str).str.replace(" ", "")
                df_clientes_unicos = df[(~df['ES_UTE']) & (df[col_grupo] == grupo_seleccionado)][[col_cif, col_nombre_cliente]].drop_duplicates()
                df_clientes_unicos[col_nombre_cliente] = df_clientes_unicos[col_nombre_cliente].fillna("").str.strip()
                df_clientes_unicos[col_cif] = df_clientes_unicos[col_cif].fillna("").str.strip()
                df_clientes_unicos = df_clientes_unicos.sort_values(col_nombre_cliente)

                opciones_clientes = ["(Todos los clientes del grupo)"] + [
                    f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
                    for _, row in df_clientes_unicos.iterrows()
                ]

                cliente_final_display = st.selectbox("Selecciona cliente final (opcional)", opciones_clientes)

                # Filtrar facturas según selección
                if cliente_final_display == "(Todos los clientes del grupo)":
                    df_filtrado = df[df[col_grupo] == grupo_seleccionado].copy()
                else:
                    cliente_final_cif = cliente_final_display.split(" - ")[0].replace(" ", "")
                    df_filtrado = df[df[col_cif] == cliente_final_cif].copy()

         # Filtrar solo facturas de TSS
                df_tss = df_filtrado[df_filtrado[col_sociedad] == 'TSS']
                if df_tss.empty:
                    st.warning("⚠️ No se encontraron facturas de TSS (90) en la selección")
                else:
                    facturas_cliente = df_tss[[col_factura, col_fecha_emision, 'IMPORTE_CORRECTO']].dropna()
                    facturas_cliente = facturas_cliente.sort_values('IMPORTE_CORRECTO', ascending=False)

                    opciones_facturas = [
                        f"{row[col_factura]} - {row[col_fecha_emision].date()} - {row['IMPORTE_CORRECTO']:,.2f} €"
                        for _, row in facturas_cliente.iterrows()
                    ]

                    factura_final_display = st.selectbox("Selecciona factura final TSS (90)", opciones_facturas)
                    factura_final_id = factura_final_display.split(" - ")[0]
                    factura_final = df_tss[df_tss[col_factura] == factura_final_id].iloc[0]

                    st.info(f"Factura final seleccionada: **{factura_final[col_factura]}** "
                            f"({factura_final['IMPORTE_CORRECTO']:,.2f} €)")
                    
    # ==========================================
    # 🔹 Filtrado UTES y ejecución del solver
    # ==========================================
    grupo_seleccionado = globals().get("grupo_seleccionado", None)

    if grupo_seleccionado:  # Solo si hay un grupo válido
        grupo_filtrado = str(grupo_seleccionado).replace(" ", "")
        df[col_grupo] = df[col_grupo].astype(str).str.replace(" ", "")

        # Filtrar UTES del mismo grupo y eliminar negativas
        df_utes_grupo = df[
            (df[col_grupo] == grupo_filtrado) &
            (df['ES_UTE'])
        ].copy()

        df_utes_grupo = df_utes_grupo[df_utes_grupo['IMPORTE_CORRECTO'].fillna(0) > 0]

        if df_utes_grupo.empty:
            st.warning("⚠️ No hay UTES válidas (positivas) para esta selección")
        else:
            # Preparar opciones de selección de socios
            df_utes_unicos = df_utes_grupo[[col_cif, col_nombre_cliente]].drop_duplicates().sort_values(by=col_cif)
            opciones_utes = [
                f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
                for _, row in df_utes_unicos.iterrows()
            ]
            mapping_utes_cif = dict(zip(opciones_utes, df_utes_unicos[col_cif]))

            socios_display = st.multiselect("Selecciona CIF(s) de la UTE (socios)", opciones_utes, key="multiselect_socios_utes")
            socios_cifs = [mapping_utes_cif[s] for s in socios_display]

            df_internas = df_utes_grupo[df_utes_grupo[col_cif].isin(socios_cifs)].copy()

                       
            # ==========================================
            # 🔹 1) Cuadrar TSS con internas (opcional)
            # ==========================================
            df_resultado_tss = pd.DataFrame()
            if not df_tss_selec.empty:
                
                resultados_internas = []
                used_interna_idxs = set()  # control global de internas ya usadas

                for _, tss_row in df_tss_selec.iterrows():
                    df_internas_available = df_internas[~df_internas.index.isin(used_interna_idxs)].copy()
                    if df_internas_available.empty:
                        continue
                # 🔹 0) PRISMA para esta TSS (90)
                    prisma_cubierto, pendiente_prisma = hook_prisma(
                        tss_row,
                        df_prisma,
                        col_num_factura_prisma,
                        col_cif_prisma,
                        col_importe_prisma,
                        col_id_ute_prisma
                    )

                    if prisma_cubierto:
                        st.info(f"🟢 TSS {tss_row[col_factura]} cubierta por PRISMA — se omite COBRA")
                        continue   # ⛔ NO pasar por cuadrar_internas

                    # 🔹 1) Si PRISMA no cubre, ir a COBRA
                    df_internas_available = df_internas[~df_internas.index.isin(used_interna_idxs)].copy()
                    if df_internas_available.empty:
                        continue
                    df_cuadras = cuadrar_internas(tss_row, df_internas_available)
                    if df_cuadras is None or df_cuadras.empty:
                        continue

                    try:
                        idx_col_doc = df_cuadras.columns.get_loc(col_factura)
                        df_cuadras.insert(idx_col_doc, "TSS_90", tss_row[col_factura])
                    except Exception:
                        df_cuadras["TSS_90"] = tss_row[col_factura]

                    resultados_internas.append(df_cuadras)
                    used_interna_idxs.update(df_cuadras.index.tolist())

                if resultados_internas:
                    df_resultado_tss = pd.concat(resultados_internas, ignore_index=False)
                    if col_sociedad in df_resultado_tss.columns and col_factura in df_resultado_tss.columns:
                        df_resultado_tss = df_resultado_tss.drop_duplicates(subset=[col_sociedad, col_factura])
                    st.success("✅ Se cuadraron las TSS con las internas")
                    st.dataframe(df_resultado_tss, use_container_width=True)
                    

            # ==========================================
            # 🔹 2) Cuadrar factura final con internas
            # ==========================================
            
            df_resultado_factura = pd.DataFrame()

            # 🔹 Chivato PRISMA
            if factura_final is not None and not df_prisma.empty:
                prisma_cubierto, pendiente_prisma = hook_prisma(
                    factura_final,
                    df_prisma,
                    col_num_factura_prisma,
                    col_cif_prisma,
                    col_importe_prisma,
                    col_id_ute_prisma
                )

                           
            if factura_final is not None and not df_internas.empty:
                df_resultado_factura = cuadrar_internas(factura_final, df_internas)
                if df_resultado_factura.empty:
                    st.warning("❌ No se encontró combinación de facturas internas que cuadre con la factura externa")
                else:
                    st.success(f"✅ Se han seleccionado {len(df_resultado_factura)} factura(s) interna(s) que cuadran con la externa")
                    st.dataframe(df_resultado_factura[[col_factura, col_cif, col_nombre_cliente,
                                                    'IMPORTE_CORRECTO', col_fecha_emision, col_sociedad]],
                                use_container_width=True)

            # ==========================================
            # 🔹 3) Determinar el DataFrame final a usar
            # ==========================================
            if not df_resultado_tss.empty:
                df_resultado = df_resultado_tss.copy()
            elif not df_resultado_factura.empty:
                df_resultado = df_resultado_factura.copy()
            else:
                df_resultado = pd.DataFrame()

            # Mostrar aviso si no hay facturas internas seleccionadas
            if df_resultado.empty:
                st.info("ℹ️ No hay facturas internas seleccionadas para intentar cuadre con pagos.")

            else:
                # Aquí NO SE PONE importe_total_final — se pone más adelante antes del Excel de pagos
                pass
            # --- asegurar importe_total_final definido ---
            if 'importe_total_final' not in locals():
                try:
                    importe_total_final = float(df_resultado['IMPORTE_CORRECTO'].sum())
                except Exception:
                    importe_total_final = 0.0


    # --- 2) leer/normalizar cobros ---
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
    if df_resultado.empty:
        st.info("ℹ️ No hay facturas internas seleccionadas para intentar cuadre con pagos.")
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

        # --- 3) preparar referencia: id factura final, fecha y importe total ---
        # obtener id y fecha de la factura final (manejamos Series o DataFrame-row)
        try:
            if isinstance(factura_final, pd.Series):
                fact_final_id = str(factura_final[col_factura])
                fecha_ref = factura_final[col_fecha_emision]
            else:
                fact_final_id = str(factura_final.iloc[0][col_factura])
                fecha_ref = factura_final.iloc[0][col_fecha_emision]
        except Exception:
            # fallback robusto
            fact_final_id = str(factura_final.get(col_factura, '')) if hasattr(factura_final, 'get') else ''
            fecha_ref = factura_final.get(col_fecha_emision, pd.NaT) if hasattr(factura_final, 'get') else pd.NaT

        # importe de referencia: debe ser el importe de la FACTURA FINAL TSS
        # preferimos IMPORTE_CORRECTO si existe en df_resultado o en factura_final
        importe_total_final = None
        if 'IMPORTE_CORRECTO' in df_resultado.columns:
            importe_total_final = float(pd.to_numeric(df_resultado['IMPORTE_CORRECTO'].sum(), errors='coerce') or 0.0)
        elif 'importe_correcto' in df_resultado.columns:
            importe_total_final = float(pd.to_numeric(df_resultado['importe_correcto'].sum(), errors='coerce') or 0.0)
        else:
            # intentar leer importe de factura_final (columna detectada antes)
            col_importe_factura = None
            posibles_importes = ['IMPORTE_CORRECTO', 'Importe', 'importe', 'TOTAL', 'total']
            for p in posibles_importes:
                if hasattr(factura_final, 'get') and factura_final.get(p) is not None:
                    col_importe_factura = p
                    break
                if not isinstance(factura_final, pd.Series) and p in factura_final.columns:
                    col_importe_factura = p
                    break
            try:
                if isinstance(factura_final, pd.Series) and col_importe_factura:
                    importe_total_final = float(factura_final[col_importe_factura])
                elif col_importe_factura:
                    importe_total_final = float(factura_final.iloc[0][col_importe_factura])
                else:
                    importe_total_final = 0.0
            except Exception:
                importe_total_final = 0.0

        
        # --- 4) normalizar lista de socios CIF que vinieron del selector (socios_cifs) ---
        try:
            socios_list = [s.replace(' ', '').upper() for s in socios_cifs]  # variable creada arriba en tu script
        except Exception:
            # fallback: extraer CIFs de df_resultado si existe columna t_doc_n_m_doc o col_cif
            if 't_doc_n_m_doc' in df_resultado.columns:
                socios_list = df_resultado['t_doc_n_m_doc'].astype(str).fillna('').str.replace(' ', '').str.upper().unique().tolist()
            elif col_cif in df_resultado.columns:
                socios_list = df_resultado[col_cif].astype(str).fillna('').str.replace(' ', '').str.upper().unique().tolist()
            else:
                socios_list = []
    
        # tolerance en euros
        TOLERANCIA = 1.0

        # --- auxiliar: elegir candidato más cercano por fecha ---
        # --- auxiliar: elegir candidato más cercano por fecha (solo pagos posteriores o iguales) ---
        def choose_closest_by_date(cand_df, fecha_ref_local):
            if cand_df is None or cand_df.empty:
                return None

            tmp = cand_df.copy()
            fecha_ref_dt = pd.to_datetime(fecha_ref_local, errors='coerce')

            if 'fec_operacion' in tmp.columns:
                tmp['fec_operacion'] = pd.to_datetime(tmp['fec_operacion'], errors='coerce')
                # Filtrar solo pagos posteriores o iguales a fecha_ref
                tmp = tmp[tmp['fec_operacion'] >= fecha_ref_dt]

            # Filtrar solo filas con importe válido
            if 'importe' in tmp.columns:
                tmp = tmp[tmp['importe'].notna()]

            if tmp.empty:
                return None

            # Elegir pago más cercano posterior (min diferencia)
            if 'fec_operacion' in tmp.columns and tmp['fec_operacion'].notna().any():
                tmp['diff'] = (tmp['fec_operacion'] - fecha_ref_dt).dt.total_seconds()
                chosen = tmp.sort_values('diff').iloc[0]
            else:
                # Si no hay fechas, coger el primero disponible
                chosen = tmp.iloc[0]

            return chosen.to_dict()


        pago_elegido = None

        # --- Paso A: buscar por posible_factura EXACTA + importe total dentro de tolerancia
        if not df_cobros.empty and fact_final_id:
            cand_pf = df_cobros[df_cobros.get('posible_factura', '').astype(str) == fact_final_id].copy()
            if not cand_pf.empty and 'importe' in cand_pf.columns:
                cand_pf = cand_pf[cand_pf['importe'].notna()]
                cand_pf = cand_pf[(cand_pf['importe'] >= (importe_total_final - TOLERANCIA)) &
                                (cand_pf['importe'] <= (importe_total_final + TOLERANCIA))]
                if not cand_pf.empty:
                    pago_elegido = choose_closest_by_date(cand_pf, fecha_ref)

        # --- Paso B: si no hay, buscar por IMPORTE + CIF (CIF debe pertenecer a socios_list)
        if pago_elegido is None and not df_cobros.empty:
            # detectar columna de CIF/NIF en df_cobros
            cif_col = None
            for c in df_cobros.columns:
                if any(k in c for k in ['cif', 'nif', 'titular', 'benef', 'beneficiario', 'cliente', 'titular_nif']):
                    cif_col = c
                    break

            candidatos = df_cobros.copy()
            if 'importe' in candidatos.columns:
                candidatos = candidatos[candidatos['importe'].notna()]
                candidatos = candidatos[(candidatos['importe'] >= (importe_total_final - TOLERANCIA)) &
                                        (candidatos['importe'] <= (importe_total_final + TOLERANCIA))]
            else:
                candidatos = candidatos.iloc[0:0]

            if cif_col and socios_list:
                candidatos[cif_col] = candidatos[cif_col].astype(str).fillna('').str.replace(' ', '').str.upper()
                candidatos_por_cif = candidatos[candidatos[cif_col].isin(socios_list)].copy()
                if not candidatos_por_cif.empty:
                    # priorizamos posible_factura dentro de este subset
                    pf_match = candidatos_por_cif[candidatos_por_cif.get('posible_factura','').astype(str) == fact_final_id]
                    if not pf_match.empty:
                        pago_elegido = choose_closest_by_date(pf_match, fecha_ref)
                    else:
                        # fallback: por fecha
                        pago_elegido = choose_closest_by_date(candidatos_por_cif, fecha_ref)

            
        # --- 5) asignar UNICO pago encontrado (si existe) a TODO df_resultado ---
        # inicializamos columnas de pago en df_resultado
        df_resultado.loc[:, 'posible_pago'] = 'No'
        df_resultado.loc[:, 'pagos_detalle'] = None
        df_resultado.loc[:, 'Pago_Importe'] = pd.NA
        df_resultado.loc[:, 'Pago_Fecha'] = pd.NaT
        df_resultado.loc[:, 'Pago_Norma43'] = pd.NA
        df_resultado.loc[:, 'Pago_CIF'] = pd.NA

        if pago_elegido is not None:
            p = pago_elegido
            importe_pago = p.get('importe') if p.get('importe') is not None else 0.0
            fecha_pago = p.get('fec_operacion') if 'fec_operacion' in p else None
            norma_pago = p.get('norma_43') if 'norma_43' in p else ''

            # intentar extraer cif si detectamos columna cif_col
            cif_pago_text = ''
            try:
                if 'cif_col' in locals() and cif_col in p:
                    cif_pago_text = p.get(cif_col, '')
            except Exception:
                cif_pago_text = ''

            resumen = f"Pago: {float(importe_pago):.2f} € ({pd.to_datetime(fecha_pago, errors='coerce').date() if pd.notna(fecha_pago) else ''}) Norma43: {norma_pago} CIF: {cif_pago_text}"
            df_resultado.loc[:, 'posible_pago'] = 'Sí'
            df_resultado.loc[:, 'pagos_detalle'] = resumen
            df_resultado.loc[:, 'Pago_Importe'] = importe_pago
            df_resultado.loc[:, 'Pago_Fecha'] = fecha_pago
            df_resultado.loc[:, 'Pago_Norma43'] = norma_pago
            if cif_pago_text:
                df_resultado.loc[:, 'Pago_CIF'] = cif_pago_text

            st.success(f"✅ Pago encontrado y asignado al total: {float(importe_pago):.2f} € (Factura final: {fact_final_id})")
        else:
            st.info("⚠️ No se encontró un pago único que cuadre con la factura final según la lógica solicitada.")

        # --- 6) mostrar tabla final con info de pago ---

        # --- columnas base de df_resultado ---
        columnas_base = [col_factura, col_cif, col_nombre_cliente, 'IMPORTE_CORRECTO', col_fecha_emision, col_sociedad]
        columnas_base = [c for c in columnas_base if c in df_resultado.columns]

        # --- columnas de pago ---
        columnas_pago = [c for c in df_resultado.columns if c.lower().startswith('pago') or c in ['posible_pago', 'pagos_detalle']]

        # --- añadir info de factura final ---
        df_resultado['Factura_Final'] = fact_final_id
        df_resultado['Fecha_Factura_Final'] = fecha_ref
        df_resultado['Importe_Factura_Final'] = importe_total_final

        # --- quitar duplicados ---
        df_resultado = df_resultado.loc[:, ~df_resultado.columns.duplicated()]

        # --- definir columnas finales con factura final primero ---
        columnas_finales = ['Factura_Final', 'Fecha_Factura_Final', 'Importe_Factura_Final'] + columnas_base + columnas_pago
        # eliminar posibles duplicados conservando el orden
        columnas_finales = list(dict.fromkeys(columnas_finales))

        # --- mostrar en Streamlit ---
        st.dataframe(df_resultado[columnas_finales], use_container_width=True)

        # --- 7) descargar ---
        def to_excel(df_out):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_out.to_excel(writer, index=False, sheet_name="Resultado")
            return output.getvalue()

        excel_data = to_excel(df_resultado[columnas_finales])
        st.download_button(
            label="📥 Descargar Excel con facturas internas seleccionadas y pagos",
            data=excel_data,
            file_name=f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if pago_elegido is not None:
            rows = []

            # --- Caso AGRUPADO: varias facturas TSS seleccionadas ---
            if isinstance(factura_final, pd.Series) and factura_final.get(col_cif) == "AGRUPADO":

                link_col = 'TSS_90' if 'TSS_90' in df_resultado.columns else col_factura

                for _, tss_row in df_tss_selec.iterrows():
                    tss_num = tss_row[col_factura]

                    if link_col in df_resultado.columns:
                        socios_factura = df_resultado[df_resultado[link_col] == tss_num].copy()
                    else:
                        socios_factura = df_resultado[df_resultado[col_factura] == tss_num].copy()

                    if socios_factura.empty:
                        continue

                    if col_sociedad in socios_factura.columns:
                        socios_unicos = socios_factura.drop_duplicates(subset=[col_sociedad])
                    else:
                        socios_unicos = socios_factura.drop_duplicates()

                    for _, socio in socios_unicos.iterrows():
                        rows.append({
                            "GESTOR DE COBROS": pago_elegido.get("gestor_de_cobros", ""),
                            "NOMBRE UTE": " ".join(df_resultado[col_nombre_cliente].unique()) if col_nombre_cliente in df_resultado.columns else "",
                            "CIF UTE": " - ".join(df_resultado[col_cif].unique()) if col_cif in df_resultado.columns else "",
                            "FECHA COBRO": pd.to_datetime(pago_elegido.get("fec_operacion")).strftime("%d/%m/%Y") 
                                        if pago_elegido.get("fec_operacion") is not None else "",
                            "IMPORTE TOTAL COBRADO": pago_elegido.get("importe", 0.0),
                            "CIF CLIENTE": tss_row.get(col_cif, ""),
                            "NOMBRE CLIENTE": tss_row.get(col_nombre_cliente, ""),
                            "FECHA FRA. UTE (de la ute a cliente final)": pd.to_datetime(tss_row.get(col_fecha_emision)).strftime("%d/%m/%Y")
                                                                        if pd.notna(tss_row.get(col_fecha_emision)) else "",
                            "Nº FRA. UTE (de la ute a cliente final)": tss_row.get(col_factura, ""),
                            "IMPORTE FRA. UTE (de la ute a cliente final)": tss_row.get("IMPORTE_CORRECTO", 0.0),
                            "FECHA FRA. DEL SOCIO (RR,ADM,TSOL)": pd.to_datetime(socio.get(col_fecha_emision)).strftime("%d/%m/%Y") 
                                                                if pd.notna(socio.get(col_fecha_emision)) else "",
                            "NºFRA. DEL SOCIO (RR,ADM,TSOL)": socio.get(col_factura, ""),
                            "IMPORTE FRA. DEL SOCIO (RR,ADM,TSOL)": socio.get("IMPORTE_CORRECTO", 0.0),
                            "SOCIO A PAGAR": socio.get(col_sociedad, ""),
                            "ID MOVIMIENTO": pago_elegido.get("id_movimiento", ""),
                        })

            else:
                # --- Caso normal: solo una factura final ---
                for _, socio in df_resultado.iterrows():
                    rows.append({
                        "GESTOR DE COBROS": pago_elegido.get("gestor_de_cobros", ""),
                        "NOMBRE UTE": " ".join(df_resultado[col_nombre_cliente].unique()) if col_nombre_cliente in df_resultado.columns else "",
                        "CIF UTE": " - ".join(df_resultado[col_cif].unique()) if col_cif in df_resultado.columns else "",
                        "FECHA COBRO": pd.to_datetime(pago_elegido.get("fec_operacion")).strftime("%d/%m/%Y") 
                                    if pago_elegido.get("fec_operacion") is not None else "",
                        "IMPORTE TOTAL COBRADO": pago_elegido.get("importe", 0.0),
                        "CIF CLIENTE": factura_final.get(col_cif, "") if isinstance(factura_final, pd.Series) else factura_final[col_cif],
                        "NOMBRE CLIENTE": factura_final.get(col_nombre_cliente, "") if isinstance(factura_final, pd.Series) else factura_final[col_nombre_cliente],
                        "FECHA FRA. UTE (de la ute a cliente final)": pd.to_datetime(factura_final.get(col_fecha_emision)).strftime("%d/%m/%Y")
                                                                    if isinstance(factura_final, pd.Series) and pd.notna(factura_final.get(col_fecha_emision))
                                                                    else (pd.to_datetime(factura_final.iloc[0][col_fecha_emision]).strftime("%d/%m/%Y") if not isinstance(factura_final, pd.Series) else ""),
                        "Nº FRA. UTE (de la ute a cliente final)": factura_final.get(col_factura, "") if isinstance(factura_final, pd.Series) else factura_final.iloc[0][col_factura],
                        "IMPORTE FRA. UTE (de la ute a cliente final)": factura_final.get("IMPORTE_CORRECTO", 0.0) if isinstance(factura_final, pd.Series) else factura_final.iloc[0].get("IMPORTE_CORRECTO", 0.0),
                        "FECHA FRA. DEL SOCIO (RR,ADM,TSOL)": pd.to_datetime(socio.get(col_fecha_emision)).strftime("%d/%m/%Y") 
                                                            if pd.notna(socio.get(col_fecha_emision)) else "",
                        "NºFRA. DEL SOCIO (RR,ADM,TSOL)": socio.get(col_factura, ""),
                        "IMPORTE FRA. DEL SOCIO (RR,ADM,TSOL)": socio.get("IMPORTE_CORRECTO", 0.0),
                        "SOCIO A PAGAR": socio.get(col_sociedad, ""),
                        "ID MOVIMIENTO": pago_elegido.get("id_movimiento", ""),
                    })

            df_carta_pago = pd.DataFrame(rows)

                
        # --- 8) generar carta de pago ---
            if pago_elegido is not None:
                rows = []

                # --- Caso AGRUPADO: varias facturas TSS seleccionadas ---
                if isinstance(factura_final, pd.Series) and factura_final.get(col_cif) == "AGRUPADO":

                    link_col = 'TSS_90' if 'TSS_90' in df_resultado.columns else col_factura

                    for _, tss_row in df_tss_selec.iterrows():
                        tss_num = tss_row[col_factura]

                        if link_col in df_resultado.columns:
                            socios_factura = df_resultado[df_resultado[link_col] == tss_num].copy()
                        else:
                            socios_factura = df_resultado[df_resultado[col_factura] == tss_num].copy()

                        if socios_factura.empty:
                            continue

                        if col_sociedad in socios_factura.columns:
                            socios_unicos = socios_factura.drop_duplicates(subset=[col_sociedad])
                        else:
                            socios_unicos = socios_factura.drop_duplicates()

                        for _, socio in socios_unicos.iterrows():
                            rows.append({
                                "GESTOR DE COBROS": pago_elegido.get("gestor_de_cobros", ""),
                                "NOMBRE UTE": " ".join(df_resultado[col_nombre_cliente].unique()) if col_nombre_cliente in df_resultado.columns else "",
                                "CIF UTE": " - ".join(df_resultado[col_cif].unique()) if col_cif in df_resultado.columns else "",
                                "FECHA COBRO": pd.to_datetime(pago_elegido.get("fec_operacion")).strftime("%d/%m/%Y") 
                                            if pago_elegido.get("fec_operacion") is not None else "",
                                "IMPORTE TOTAL COBRADO": pago_elegido.get("importe", 0.0),
                                "CIF CLIENTE": tss_row.get(col_cif, ""),
                                "NOMBRE CLIENTE": tss_row.get(col_nombre_cliente, ""),
                                "FECHA FRA. UTE (de la ute a cliente final)": pd.to_datetime(tss_row.get(col_fecha_emision)).strftime("%d/%m/%Y")
                                                                            if pd.notna(tss_row.get(col_fecha_emision)) else "",
                                "Nº FRA. UTE (de la ute a cliente final)": tss_row.get(col_factura, ""),
                                "IMPORTE FRA. UTE (de la ute a cliente final)": tss_row.get("IMPORTE_CORRECTO", 0.0),
                                "FECHA FRA. DEL SOCIO (RR,ADM,TSOL)": pd.to_datetime(socio.get(col_fecha_emision)).strftime("%d/%m/%Y") 
                                                                    if pd.notna(socio.get(col_fecha_emision)) else "",
                                "NºFRA. DEL SOCIO (RR,ADM,TSOL)": socio.get(col_factura, ""),
                                "IMPORTE FRA. DEL SOCIO (RR,ADM,TSOL)": socio.get("IMPORTE_CORRECTO", 0.0),
                                "SOCIO A PAGAR": socio.get(col_sociedad, ""),
                                "ID MOVIMIENTO": pago_elegido.get("id_movimiento", ""),
                            })

                else:
                    # --- Caso normal: solo una factura final ---
                    for _, socio in df_resultado.iterrows():
                        rows.append({
                            "GESTOR DE COBROS": pago_elegido.get("gestor_de_cobros", ""),
                            "NOMBRE UTE": " ".join(df_resultado[col_nombre_cliente].unique()) if col_nombre_cliente in df_resultado.columns else "",
                            "CIF UTE": " - ".join(df_resultado[col_cif].unique()) if col_cif in df_resultado.columns else "",
                            "FECHA COBRO": pd.to_datetime(pago_elegido.get("fec_operacion")).strftime("%d/%m/%Y") 
                                        if pago_elegido.get("fec_operacion") is not None else "",
                            "IMPORTE TOTAL COBRADO": pago_elegido.get("importe", 0.0),
                            "CIF CLIENTE": factura_final.get(col_cif, "") if isinstance(factura_final, pd.Series) else factura_final[col_cif],
                            "NOMBRE CLIENTE": factura_final.get(col_nombre_cliente, "") if isinstance(factura_final, pd.Series) else factura_final[col_nombre_cliente],
                            "FECHA FRA. UTE (de la ute a cliente final)": pd.to_datetime(factura_final.get(col_fecha_emision)).strftime("%d/%m/%Y")
                                                                        if isinstance(factura_final, pd.Series) and pd.notna(factura_final.get(col_fecha_emision))
                                                                        else (pd.to_datetime(factura_final.iloc[0][col_fecha_emision]).strftime("%d/%m/%Y") if not isinstance(factura_final, pd.Series) else ""),
                            "Nº FRA. UTE (de la ute a cliente final)": factura_final.get(col_factura, "") if isinstance(factura_final, pd.Series) else factura_final.iloc[0][col_factura],
                            "IMPORTE FRA. UTE (de la ute a cliente final)": factura_final.get("IMPORTE_CORRECTO", 0.0) if isinstance(factura_final, pd.Series) else factura_final.iloc[0].get("IMPORTE_CORRECTO", 0.0),
                            "FECHA FRA. DEL SOCIO (RR,ADM,TSOL)": pd.to_datetime(socio.get(col_fecha_emision)).strftime("%d/%m/%Y") 
                                                                if pd.notna(socio.get(col_fecha_emision)) else "",
                            "NºFRA. DEL SOCIO (RR,ADM,TSOL)": socio.get(col_factura, ""),
                            "IMPORTE FRA. DEL SOCIO (RR,ADM,TSOL)": socio.get("IMPORTE_CORRECTO", 0.0),
                            "SOCIO A PAGAR": socio.get(col_sociedad, ""),
                            "ID MOVIMIENTO": pago_elegido.get("id_movimiento", ""),
                        })

                df_carta_pago = pd.DataFrame(rows)

                # --- Exportación a Excel ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_carta_pago.to_excel(writer, index=False, sheet_name="Carta de Pago")

                st.download_button(
                    label="📥 Descargar Carta de Pago",
                    data=output.getvalue(),
                    file_name="Carta_de_Pago.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
