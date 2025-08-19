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

# --------- Inicializar variables globales ---------
factura_final = None
df_internas = pd.DataFrame()

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
    col_grupo         = find_col(df, ['CIF_GRUPO', 'GRUPO', 'CIF Grupo'])

    faltan = []
    if not col_fecha_emision: faltan.append("fecha emisión")
    if not col_factura:       faltan.append("nº factura")
    if not col_importe:       faltan.append("importe")
    if not col_cif:           faltan.append("CIF")
    if not col_grupo:         faltan.append("CIF grupo")
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
    df['ES_UTE'] = df[col_cif].str.replace(" ", "").str.contains(r"L-00U")

    # --- Opciones de clientes finales (no UTES) ---
    df_clientes_unicos = df[~df['ES_UTE']][[col_cif, col_nombre_cliente, col_grupo]].drop_duplicates()
    df_clientes_unicos[col_nombre_cliente] = df_clientes_unicos[col_nombre_cliente].fillna("").str.strip()
    df_clientes_unicos[col_cif] = df_clientes_unicos[col_cif].fillna("").str.strip()
    df_clientes_unicos = df_clientes_unicos.sort_values(col_nombre_cliente)
    opciones_clientes = [
        f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
        for _, row in df_clientes_unicos.iterrows()
    ]
    mapping_cif = dict(zip(opciones_clientes, df_clientes_unicos[col_cif]))
    mapping_grupo = dict(zip(df_clientes_unicos[col_cif], df_clientes_unicos[col_grupo]))

    # --- Selección de cliente final ---
    cliente_final_display = st.selectbox("Selecciona cliente final (CIF - Nombre)", opciones_clientes)
    cliente_final_cif = mapping_cif[cliente_final_display]
    cliente_final_grupo = mapping_grupo[cliente_final_cif]
    df_cliente_final = df[df[col_cif] == cliente_final_cif].copy()

    # --- Filtrar solo facturas de TSS ---
    df_tss = df_cliente_final[df_cliente_final[col_sociedad] == 'TSS']
    if df_tss.empty:
        st.warning("⚠️ No se encontraron facturas de TSS para este cliente final")

    # --- Selección de factura final (TSS) ---
    facturas_cliente = df_tss[[col_factura, col_fecha_emision, 'IMPORTE_CORRECTO']].dropna()
    if not facturas_cliente.empty:
        opciones_facturas = [
            f"{row[col_factura]} - {row[col_fecha_emision].date()} - {row['IMPORTE_CORRECTO']:,.2f} €"
            for _, row in facturas_cliente.iterrows()
        ]
        factura_final_display = st.selectbox("Selecciona factura final TSS (90)", opciones_facturas)
        factura_final_id = factura_final_display.split(" - ")[0]
        factura_final = df_tss[df_tss[col_factura] == factura_final_id].iloc[0]

        st.info(f"Factura final seleccionada: **{factura_final[col_factura]}** "
                f"({factura_final['IMPORTE_CORRECTO']:,.2f} €)")
    else:
        st.warning("⚠️ No hay facturas TSS disponibles para seleccionar")
        factura_final = None

    # --- Filtrar UTES del mismo grupo y eliminar negativas ---
    df_utes_grupo = df[
        (df[col_grupo] == cliente_final_grupo) & (df['ES_UTE'])
    ].copy()

    # Eliminar importes negativos o cero
    df_utes_grupo = df_utes_grupo[df_utes_grupo['IMPORTE_CORRECTO'].fillna(0) > 0]

    if df_utes_grupo.empty:
        st.warning("⚠️ No hay UTES válidas (positivas) para este cliente final")
    else:
        # Crear lista de socios únicos para el selector
        df_utes_unicos = df_utes_grupo[[col_cif, col_nombre_cliente]].drop_duplicates().sort_values(by=col_cif)
        opciones_utes = [
            f"{row[col_cif]} - {row[col_nombre_cliente]}" if row[col_nombre_cliente] else f"{row[col_cif]}"
            for _, row in df_utes_unicos.iterrows()
        ]
        mapping_utes_cif = dict(zip(opciones_utes, df_utes_unicos[col_cif]))

        socios_display = st.multiselect("Selecciona CIF(s) de la UTE (socios)", opciones_utes)
        socios_cifs = [mapping_utes_cif[s] for s in socios_display]

        # Filtrar DataFrame interno final para el solver
        df_internas = df_utes_grupo[df_utes_grupo[col_cif].isin(socios_cifs)].copy()

        # --- Solver ---
        def cuadrar_internas(externa, df_internas, tol=100):
            """tol en céntimos, default 1€ = 100 céntimos"""
            if externa is None or df_internas.empty:
                return pd.DataFrame()

            objetivo = int(externa['IMPORTE_CENT'])
            fecha_ref = externa[col_fecha_emision]

            # Preparar datos
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

            # Suma con tolerancia
            model.Add(sum(x[i] * data[i][1] for i in range(n)) >= objetivo - tol)
            model.Add(sum(x[i] * data[i][1] for i in range(n)) <= objetivo + tol)

            # Restricción: solo una factura por sociedad
            sociedades = set(d[3] for d in data)
            for s in sociedades:
                indices = [i for i, d in enumerate(data) if d[3] == s]
                if indices:
                    model.Add(sum(x[i] for i in indices) <= 1)

            # Minimizar número de facturas y diferencia de fechas
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
            
# ----------- Resultado y descarga -----------
if factura_final is not None and not df_internas.empty:
    # --- Normalizar columnas de df_internas ---
    df_internas.columns = (
        df_internas.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[^0-9a-z]', '_', regex=True)
        .str.replace(r'__+', '_', regex=True)
        .str.strip('_')
    )

    # --- Verificar columna de CIF ---
    if 't_doc_n_m_doc' not in df_internas.columns:
        st.error("❌ No se encontró la columna de CIF (t_doc_n_m_doc) en el archivo de facturas internas.")
    else:
        # --- Selección de CIF(s) ---
        cif_seleccionados = st.multiselect(
            "Selecciona CIF(s) de la UTE (socios)",
            options=df_internas['t_doc_n_m_doc'].unique()
        )

        if cif_seleccionados:
            # --- Filtrar por CIF ---
            df_internas_filtrado = df_internas[df_internas['t_doc_n_m_doc'].isin(cif_seleccionados)]

            # --- Detectar automáticamente la columna de importe en factura_final ---
            if isinstance(factura_final, pd.Series):
                cols_factura = factura_final.index.tolist()
            else:
                cols_factura = factura_final.columns.tolist()

            st.write("📄 Columnas en factura_final:", cols_factura)

            col_importe_factura = None
            for posible in ['Importe', 'importe_correcto', 'importe', 'total', 'importe_total']:
                if posible in cols_factura:
                    col_importe_factura = posible
                    break

            if not col_importe_factura:
                st.error("❌ No se encontró ninguna columna de importe en factura_final")
            else:
                if isinstance(factura_final, pd.Series):
                    importe_factura_final = factura_final[col_importe_factura]
                else:
                    importe_factura_final = factura_final[col_importe_factura].iloc[0]

                # --- Filtrar por importe ±1€ ---
                TOLERANCIA = 1.0
                df_internas_filtrado = df_internas_filtrado[
                    df_internas_filtrado['importe_correcto'].between(
                        importe_factura_final - TOLERANCIA,
                        importe_factura_final + TOLERANCIA
                    )
                ]

                if df_internas_filtrado.empty:
                    st.warning("❌ No se encontró combinación de facturas internas que cuadre con la factura externa")
                else:
                    st.success(f"✅ Se han seleccionado {len(df_internas_filtrado)} factura(s) interna(s) que cuadran con la externa")

                    df_resultado = df_internas_filtrado.copy()
                                          
                # --- Carga opcional de pagos ---
                cobros_file = st.file_uploader("Sube el Excel de Gestor de Cobros (opcional)", type=['.xlsm', '.csv'], key="cobros")
                df_cobros = pd.DataFrame()
                if cobros_file:
                    try:
                        if cobros_file.name.endswith('.xlsm'):
                            df_cobros = pd.read_excel(cobros_file, sheet_name='Cruce_Movs', engine='openpyxl')
                        else:
                            df_cobros = pd.read_csv(cobros_file)
                    except Exception as e:
                        st.error(f"Error al leer el archivo de cobros: {e}")

                    if not df_cobros.empty:
                        # --- Normalización robusta de columnas ---
                        df_cobros.columns = (
                            df_cobros.columns
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

                        # --- Mapear columnas críticas ---
                        col_mapping = {
                            'fec_operacion': ['fec_operacion', 'fecha_operacion'],
                            'importe': ['importe', 'imp', 'monto'],
                            'norma_43': ['norma_43', 'norma43'],
                            'posible_factura': ['posible_factura', 'factura']
                        }

                        for target, possibles in col_mapping.items():
                            found = False
                            for col in possibles:
                                if col in df_cobros.columns:
                                    df_cobros.rename(columns={col: target}, inplace=True)
                                    found = True
                                    break
                            if not found:
                                for col in df_cobros.columns:
                                    if target.split('_')[0] in col:
                                        df_cobros.rename(columns={col: target}, inplace=True)
                                        break

                        # --- Verificar columnas esenciales ---
                        required_cols = ['fec_operacion', 'importe', 'norma_43', 'posible_factura']
                        missing_cols = [col for col in required_cols if col not in df_cobros.columns]
                        if missing_cols:
                            st.error(f"❌ Faltan columnas esenciales en el archivo de cobros: {missing_cols}")
                            df_cobros = pd.DataFrame()
                        else:
                            # --- Convertir tipos ---
                            df_cobros['fec_operacion'] = pd.to_datetime(df_cobros['fec_operacion'], errors='coerce')
                            df_cobros['importe'] = pd.to_numeric(df_cobros['importe'], errors='coerce')
                            df_cobros['norma_43'] = df_cobros['norma_43'].astype(str).str.strip()
                            df_cobros['posible_factura'] = df_cobros['posible_factura'].astype(str).str.strip()

                # --- Preparar columnas de pagos ---
                df_resultado['posible_pago'] = 'No'
                df_resultado['pagos_detalle'] = None

                def unique_col(df, col_base):
                    col = col_base
                    i = 1
                    while col in df.columns:
                        col = f"{col_base}_{i}"
                        i += 1
                    return col

                def buscar_pagos(fila, df_cobros):
                    posibles = []

                    # 1️⃣ Match exacto Posible Factura
                    pagos_match = df_cobros[df_cobros['posible_factura'] == str(fila.get('factura', ''))]
                    for _, p in pagos_match.iterrows():
                        if abs(p['importe'] - fila.get('importe_correcto', 0)) <= TOLERANCIA:
                            posibles.append(p)

                    if posibles:
                        return posibles

                    # 2️⃣ Buscar dentro de Norma 43
                    pagos_match_norma43 = df_cobros[df_cobros['norma_43'].str.contains(str(fila.get('factura', '')), na=False)]
                    for _, p in pagos_match_norma43.iterrows():
                        if abs(p['importe'] - fila.get('importe_correcto', 0)) <= TOLERANCIA:
                            posibles.append(p)
                    if posibles:
                        return posibles

                    # 3️⃣ Buscar por Fec. Operación a partir de fecha de la factura
                    fecha_inicio = fila.get('fecha_emision', pd.Timestamp.min)
                    pagos_fecha = df_cobros[df_cobros['fec_operacion'] >= fecha_inicio].sort_values('fec_operacion')
                    for _, p in pagos_fecha.iterrows():
                        if abs(p['importe'] - fila.get('importe_correcto', 0)) <= TOLERANCIA:
                            posibles.append(p)

                    return posibles

                # --- Aplicar búsqueda de pagos ---
                if not df_cobros.empty:
                    for idx, fila in df_resultado.iterrows():
                        pagos = buscar_pagos(fila, df_cobros)
                        if pagos:
                            df_resultado.at[idx, 'posible_pago'] = 'Sí'
                            detalles = []
                            for i, p in enumerate(pagos, 1):
                                detalles.append(f"Pago{i}: {p['importe']:.2f} € ({p['fec_operacion'].date()}) Norma43: {p['norma_43']}")
                                col_importe = unique_col(df_resultado, f'Pago{i}_Importe')
                                col_fecha = unique_col(df_resultado, f'Pago{i}_Fecha')
                                col_norma43 = unique_col(df_resultado, f'Pago{i}_Norma43')
                                df_resultado.at[idx, col_importe] = p['importe']
                                df_resultado.at[idx, col_fecha] = p['fec_operacion']
                                df_resultado.at[idx, col_norma43] = p['norma_43']
                            df_resultado.at[idx, 'pagos_detalle'] = "; ".join(detalles)

                # --- Mostrar tabla final ---
                columnas_base = ['factura', 'cif', 'nombre_cliente', 'importe_correcto',
                                 'fecha_emision', 'sociedad', 'posible_pago', 'pagos_detalle']
                columnas_base = [c for c in columnas_base if c in df_resultado.columns]
                columnas_pago = [c for c in df_resultado.columns if c.lower().startswith('pago')]
                df_resultado = df_resultado.loc[:, ~df_resultado.columns.duplicated()]
                columnas_finales = list(dict.fromkeys(columnas_base + columnas_pago))

                st.dataframe(df_resultado[columnas_finales])

                # --- Botón de descarga ---
                from io import BytesIO
                from datetime import datetime

                def to_excel(df_out):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df_out.to_excel(writer, index=False, sheet_name="Resultado")
                    return output.getvalue()

                excel_data = to_excel(df_resultado)
                st.download_button(
                    label="📥 Descargar Excel con facturas internas seleccionadas y pagos",
                    data=excel_data,
                    file_name=f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


