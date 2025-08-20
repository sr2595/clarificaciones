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

    # --- Buscar combinaciones internas (solver) ---
    df_resultado = cuadrar_internas(factura_final, df_internas)
    if df_resultado.empty:
        st.warning("❌ No se encontró combinación de facturas internas que cuadre con la factura externa")
    else:
        st.success(f"✅ Se han seleccionado {len(df_resultado)} factura(s) interna(s) que cuadran con la externa")
        st.dataframe(df_resultado[[col_factura, col_cif, col_nombre_cliente,
                                   'IMPORTE_CORRECTO', col_fecha_emision, col_sociedad]])

    # --- Ahora intentar encontrar UN ÚNICO pago para la factura final ---
    # Subir / leer fichero de cobros
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
            df_cobros = pd.DataFrame()

    # Si no hay facturas internas seleccionadas, avisamos y salimos de búsqueda de pagos
    if df_resultado.empty:
        st.info("ℹ️ No hay facturas internas seleccionadas para intentar cuadre con pagos.")
    else:
        # Normalizamos y mapeamos columnas del fichero de cobros
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

            # mapeo básico de columnas que usamos
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

            # asegurar tipos
            if 'fec_operacion' in df_cobros.columns:
                df_cobros['fec_operacion'] = pd.to_datetime(df_cobros['fec_operacion'], errors='coerce')
            if 'importe' in df_cobros.columns:
                df_cobros['importe'] = pd.to_numeric(df_cobros['importe'], errors='coerce')
            df_cobros['posible_factura'] = df_cobros.get('posible_factura', pd.Series([''] * len(df_cobros))).astype(str).str.strip()
            df_cobros['norma_43'] = df_cobros.get('norma_43', pd.Series([''] * len(df_cobros))).astype(str).str.strip()

        # parámetros
        TOLERANCIA = 1.0  # euros

        # inicializar columnas de resultado
        df_resultado['posible_pago'] = 'No'
        df_resultado['pagos_detalle'] = None
        df_resultado['Pago_Importe'] = pd.NA
        df_resultado['Pago_Fecha'] = pd.NaT
        df_resultado['Pago_Norma43'] = pd.NA
        df_resultado['Pago_CIF'] = pd.NA

        # obtener id de la factura final y fecha_ref
        try:
            if isinstance(factura_final, pd.Series):
                fact_final_id = str(factura_final[col_factura])
                fecha_ref = factura_final[col_fecha_emision]
            else:
                fact_final_id = str(factura_final.iloc[0][col_factura])
                fecha_ref = factura_final.iloc[0][col_fecha_emision]
        except Exception:
            # fallback
            fact_final_id = str(factura_final.get(col_factura, '')) if hasattr(factura_final, 'get') else ''
            fecha_ref = factura_final.get(col_fecha_emision, pd.NaT) if hasattr(factura_final, 'get') else pd.NaT

        # calcular importe total de las internas seleccionadas (suma)
        importe_total_final = None
        if 'IMPORTE_CORRECTO' in df_resultado.columns:
            importe_total_final = pd.to_numeric(df_resultado['IMPORTE_CORRECTO'].sum(), errors='coerce')
        elif 'importe_correcto' in df_resultado.columns:
            importe_total_final = pd.to_numeric(df_resultado['importe_correcto'].sum(), errors='coerce')
        else:
            # fallback: intentar leer importe de factura_final
            try:
                if isinstance(factura_final, pd.Series):
                    importe_total_final = float(factura_final.get('IMPORTE_CORRECTO') or factura_final.get('Importe') or factura_final.get('importe') or 0)
                else:
                    # factura_final puede ser dataframe/row
                    importe_total_final = float(factura_final.iloc[0].get('IMPORTE_CORRECTO') or factura_final.iloc[0].get('Importe') or factura_final.iloc[0].get('importe') or 0)
            except Exception:
                importe_total_final = 0.0

        # normalizar socios CIFs: preferimos variable socios_cifs si existe, sino sacamos de df_resultado
        try:
            socios_list = [s.replace(' ', '').upper() for s in socios_cifs]  # variable creada en la parte superior del script
        except Exception:
            # fallback a los CIFs presentes en df_resultado (columna t_doc_n_m_doc o col_cif)
            if 't_doc_n_m_doc' in df_resultado.columns:
                socios_list = df_resultado['t_doc_n_m_doc'].astype(str).fillna('').apply(lambda x: x.replace(' ', '').upper()).unique().tolist()
            elif col_cif in df_resultado.columns:
                socios_list = df_resultado[col_cif].astype(str).fillna('').apply(lambda x: x.replace(' ', '').upper()).unique().tolist()
            else:
                socios_list = []

        # función auxiliar: elegir candidato más cercano por fecha
        def choose_closest_by_date(cand_df, fecha_ref):
            if cand_df.empty:
                return None
            tmp = cand_df.copy()
            if 'fec_operacion' in tmp.columns:
                tmp['fec_operacion'] = pd.to_datetime(tmp['fec_operacion'], errors='coerce')
            # preferir filas con fecha válida
            if 'fec_operacion' in tmp.columns and tmp['fec_operacion'].notna().any():
                fecha_ref_dt = pd.to_datetime(fecha_ref, errors='coerce')
                tmp = tmp[tmp['fec_operacion'].notna()].copy()
                tmp['diff'] = (tmp['fec_operacion'] - fecha_ref_dt).abs().dt.total_seconds().fillna(1e18)
                chosen = tmp.sort_values('diff').iloc[0]
            else:
                chosen = tmp.iloc[0]
            return chosen.to_dict()

        pago_elegido = None

        # --- Paso 1: buscar por posible_factura EXACTA y importe total dentro de la tolerancia ---
        if not df_cobros.empty and fact_final_id:
            candidatos_pf = df_cobros[
                df_cobros['posible_factura'].astype(str).fillna('') == fact_final_id
            ].copy()
            if not candidatos_pf.empty and 'importe' in candidatos_pf.columns:
                candidatos_pf = candidatos_pf[candidatos_pf['importe'].notna()]
                candidatos_pf = candidatos_pf[(candidatos_pf['importe'] >= (importe_total_final - TOLERANCIA)) &
                                              (candidatos_pf['importe'] <= (importe_total_final + TOLERANCIA))]
                if not candidatos_pf.empty:
                    pago_elegido = choose_closest_by_date(candidatos_pf, fecha_ref)

        # --- Paso 2: si no hay, buscar por IMPORTE + CIF (CIF debe ser uno de socios_list) ---
        if pago_elegido is None and not df_cobros.empty:
            # detectar columna potencial de CIF en df_cobros
            cif_col = None
            for c in df_cobros.columns:
                if any(k in c for k in ['cif', 'nif', 'titular', 'benef', 'beneficiario', 'cliente', 'domicilio_cliente']):
                    cif_col = c
                    break

            candidatos = df_cobros.copy()
            if 'importe' in candidatos.columns:
                candidatos = candidatos[candidatos['importe'].notna()]
                candidatos = candidatos[(candidatos['importe'] >= (importe_total_final - TOLERANCIA)) &
                                        (candidatos['importe'] <= (importe_total_final + TOLERANCIA))]
            else:
                candidatos = candidatos.iloc[0:0]  # vacío si no hay columna importe

            # si hay columna CIF y tenemos lista de socios, filtramos por ellos
            if cif_col and socios_list:
                # normalizamos valores del CIF en df_cobros
                candidatos[cif_col] = candidatos[cif_col].astype(str).fillna('').str.replace(' ', '').str.upper()
                candidatos_por_cif = candidatos[candidatos[cif_col].isin(socios_list)].copy()
                if not candidatos_por_cif.empty:
                    # prefer posible_factura exacta entre estos
                    pf_match = candidatos_por_cif[candidatos_por_cif['posible_factura'].astype(str).fillna('') == fact_final_id]
                    if not pf_match.empty:
                        pago_elegido = choose_closest_by_date(pf_match, fecha_ref)
                    else:
                        # buscar en norma_43
                        if 'norma_43' in candidatos_por_cif.columns:
                            norma_match = candidatos_por_cif[candidatos_por_cif['norma_43'].str.contains(fact_final_id, na=False)]
                            if not norma_match.empty:
                                pago_elegido = choose_closest_by_date(norma_match, fecha_ref)
                        # si no hay pf/norma que coincida, elegimos el más cercano por fecha directamente
                        if pago_elegido is None:
                            pago_elegido = choose_closest_by_date(candidatos_por_cif, fecha_ref)

            # --- Paso 3 (fallback): si no hay candidatos por CIF, intentar solo por importe en todo df_cobros
            if pago_elegido is None and not candidatos.empty:
                # priorizar pf exacto y luego norma_43, luego por fecha
                pf_match = candidatos[candidatos['posible_factura'].astype(str).fillna('') == fact_final_id]
                if not pf_match.empty:
                    pago_elegido = choose_closest_by_date(pf_match, fecha_ref)
                else:
                    if 'norma_43' in candidatos.columns:
                        norma_match = candidatos[candidatos['norma_43'].str.contains(fact_final_id, na=False)]
                        if not norma_match.empty:
                            pago_elegido = choose_closest_by_date(norma_match, fecha_ref)
                    if pago_elegido is None:
                        pago_elegido = choose_closest_by_date(candidatos, fecha_ref)

        # --- Asignar pago único (si se encontró) a todas las filas internas seleccionadas ---
        if pago_elegido is not None:
            pago = pago_elegido
            try:
                fecha_text = pd.to_datetime(pago.get('fec_operacion')).date() if pd.notna(pago.get('fec_operacion')) else ''
            except Exception:
                fecha_text = ''
            importe_text = pago.get('importe', 0)
            norma_text = pago.get('norma_43', '') if pago.get('norma_43') is not None else ''
            cif_pago_text = ''
            # si existe columna CIF detectada anteriormente en df_cobros, sacamos su valor
            try:
                if 'cif_col' in locals() and cif_col in pago:
                    cif_pago_text = pago.get(cif_col, '')
            except Exception:
                cif_pago_text = ''

            resumen = f"Pago: {importe_text:.2f} € ({fecha_text}) Norma43: {norma_text} CIF: {cif_pago_text}"
            df_resultado.loc[:, 'posible_pago'] = 'Sí'
            df_resultado.loc[:, 'pagos_detalle'] = resumen
            df_resultado.loc[:, 'Pago_Importe'] = pago.get('importe')
            df_resultado.loc[:, 'Pago_Fecha'] = pago.get('fec_operacion')
            df_resultado.loc[:, 'Pago_Norma43'] = pago.get('norma_43')
            if cif_pago_text:
                df_resultado.loc[:, 'Pago_CIF'] = cif_pago_text

            st.success(f"✅ Pago encontrado y asignado al total: {importe_text:.2f} € (Factura final: {fact_final_id})")
        else:
            st.info("⚠️ No se encontró un pago único que cuadre con la factura final según la lógica: 1) posible_factura, 2) importe+CIF (UTE socios), 3) fallback por importe.")

        # --- Mostrar tabla final con columna de pagos ---
        columnas_base = [col_factura, col_cif, col_nombre_cliente, 'IMPORTE_CORRECTO', col_fecha_emision, col_sociedad]
        columnas_base = [c for c in columnas_base if c in df_resultado.columns]
        columnas_pago = [c for c in df_resultado.columns if c.lower().startswith('pago') or c in ['posible_pago', 'pagos_detalle']]
        df_resultado = df_resultado.loc[:, ~df_resultado.columns.duplicated()]
        columnas_finales = list(dict.fromkeys(columnas_base + columnas_pago))
        st.dataframe(df_resultado[columnas_finales], use_container_width=True)

        # --- Botón de descarga ---
        from io import BytesIO
        from datetime import datetime

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
