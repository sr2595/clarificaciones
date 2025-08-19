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
    # --- Normalizar columnas de df_internas (por si vienen con mayúsculas/espacios) ---
    df_internas = df_internas.copy()
    df_internas.columns = (
        df_internas.columns
        .str.strip().str.lower()
        .str.replace(r'[^0-9a-z]', '_', regex=True)
        .str.replace(r'__+', '_', regex=True)
        .str.strip('_')
    )

    # Comprobamos que existen las columnas necesarias en df_internas
    req_internas = ['t_doc_n_m_doc', 'importe_correcto']
    faltan_internas = [c for c in req_internas if c not in df_internas.columns]
    if faltan_internas:
        st.error(f"❌ Falta(n) columna(s) en facturas internas: {faltan_internas}")
        st.stop()

    # ----> OJO: no hay multiselect aquí <----  (df_internas ya está filtrado por los CIF elegidos arriba)
    df_internas_filtrado = df_internas.copy()

    # --- Detectar el importe de la factura final (aceptamos varios nombres) ---
    if isinstance(factura_final, pd.Series):
        cols_factura = factura_final.index.tolist()
    else:
        cols_factura = factura_final.columns.tolist()

    posibles_importes = ['importe_correcto', 'importe', 'Importe', 'total', 'TOTAL', 'importe_total', 'IMPORTE_CORRECTO']
    col_importe_factura = next((c for c in posibles_importes if c in cols_factura), None)

    if not col_importe_factura:
        st.error("❌ No encuentro la columna de importe en la factura final.")
        st.stop()

    # Leer el valor del importe de la final
    if isinstance(factura_final, pd.Series):
        valor_importe = factura_final[col_importe_factura]
    else:
        valor_importe = factura_final[col_importe_factura].iloc[0]

    # Convertir a número de forma robusta (por si viene como texto con coma)
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            try:
                return float(str(v).replace('.', '').replace(',', '.'))
            except Exception:
                return None

    importe_factura_final = _to_float(valor_importe)
    if importe_factura_final is None:
        st.error("❌ Importe de la factura final no es numérico.")
        st.stop()

    # --- Filtrar internas por importe ±1€ ---
    TOLERANCIA = 1.0
    df_internas_filtrado = df_internas_filtrado[
        df_internas_filtrado['importe_correcto'].between(
            importe_factura_final - TOLERANCIA,
            importe_factura_final + TOLERANCIA
        )
    ]

    if df_internas_filtrado.empty:
        st.warning("❌ No se encontró combinación de facturas internas que cuadre con la factura externa")
        st.stop()

    st.success(f"✅ Se han seleccionado {len(df_internas_filtrado)} factura(s) interna(s) que cuadran por importe")

    df_resultado = df_internas_filtrado.copy()

    # ===================== CARGA OPCIONAL DE PAGOS =====================
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
            # Normalizar columnas cobros
            df_cobros.columns = (
                df_cobros.columns
                .str.strip().str.lower()
                .str.replace(r'[áàäâ]', 'a', regex=True)
                .str.replace(r'[éèëê]', 'e', regex=True)
                .str.replace(r'[íìïî]', 'i', regex=True)
                .str.replace(r'[óòöô]', 'o', regex=True)
                .str.replace(r'[úùüû]', 'u', regex=True)
                .str.replace(r'[^0-9a-z]', '_', regex=True)
                .str.replace(r'__+', '_', regex=True)
                .str.strip('_')
            )

            # Mapear columnas críticas (incluimos CIF en cobros para poder filtrar por los de la UTE)
            col_map = {
                'fec_operacion': ['fec_operacion', 'fecha_operacion', 'fecha_op'],
                'importe': ['importe', 'imp', 'monto', 'amount'],
                'posible_factura': ['posible_factura', 'factura', 'ref_factura'],
                'cif': ['cif', 'nif', 't_doc_n_m_doc', 't_doc_n_m__doc', 'doc_cliente', 'doc']
            }
            for target, cands in col_map.items():
                if target not in df_cobros.columns:
                    for c in cands:
                        if c in df_cobros.columns:
                            df_cobros.rename(columns={c: target}, inplace=True)
                            break

            # Tipos
            if 'fec_operacion' in df_cobros.columns:
                df_cobros['fec_operacion'] = pd.to_datetime(df_cobros['fec_operacion'], errors='coerce')
            if 'importe' in df_cobros.columns:
                df_cobros['importe'] = pd.to_numeric(df_cobros['importe'], errors='coerce')
            if 'posible_factura' in df_cobros.columns:
                df_cobros['posible_factura'] = df_cobros['posible_factura'].astype(str).str.strip()
            if 'cif' in df_cobros.columns:
                df_cobros['cif'] = df_cobros['cif'].astype(str).str.upper().str.strip()

    # ===================== MATCH DE PAGO ÚNICO (SIN NORMA 43) =====================
    df_resultado['posible_pago'] = 'No'
    df_resultado['pagos_detalle'] = None

    # CIF de las internas (para filtrar cobros por los mismos CIF de la UTE)
    cif_refs = (
        df_internas['t_doc_n_m_doc']
        .dropna().astype(str).str.upper().str.strip().unique().tolist()
        if 't_doc_n_m_doc' in df_internas.columns else []
    )

    # Elegimos un único pago que coincida en importe (±1€) y, si existe, con el mismo posible_factura.
    pago_encontrado = None
    if not df_cobros.empty and 'importe' in df_cobros.columns:
        # 1) Filtrar por CIF de la UTE si la columna existe
        candidatos = df_cobros.copy()
        if 'cif' in candidatos.columns and cif_refs:
            candidatos = candidatos[candidatos['cif'].isin(cif_refs)]

        # 2) Filtrar por importe ±1€
        candidatos = candidatos[candidatos['importe'].between(
            importe_factura_final - TOLERANCIA, importe_factura_final + TOLERANCIA
        )]

        # 3) Priorizar match por posible_factura si existe
        if 'posible_factura' in candidatos.columns:
            factura_ref = None
            # intentamos obtener la referencia de factura (columna 'factura' en df_internas si existe)
            if 'factura' in df_internas.columns and not df_internas['factura'].isna().all():
                factura_ref = str(df_internas['factura'].iloc[0])

            if factura_ref:
                preferidos = candidatos[candidatos['posible_factura'] == factura_ref]
                if not preferidos.empty:
                    candidatos = preferidos

        # Elegir el más cercano a la fecha de emisión de la factura final (si se dispone)
        # Detectar fecha emision en factura_final
        posibles_fechas = ['fecha_emision', 'fecha', 'fx_emision', 'Fecha Emisión', 'Fecha Emision']
        if isinstance(factura_final, pd.Series):
            fecha_cols = factura_final.index.tolist()
            fecha_ref = None
            for c in posibles_fechas:
                if c in fecha_cols:
                    fecha_ref = pd.to_datetime(factura_final[c], errors='coerce')
                    break
        else:
            fecha_cols = factura_final.columns.tolist()
            fecha_ref = None
            for c in posibles_fechas:
                if c in fecha_cols:
                    fecha_ref = pd.to_datetime(factura_final[c].iloc[0], errors='coerce')
                    break

        if 'fec_operacion' in candidatos.columns and not candidatos.empty:
            if pd.notna(fecha_ref):
                candidatos = candidatos.assign(
                    _diff=(candidatos['fec_operacion'] - fecha_ref).abs()
                ).sort_values('_diff')
            else:
                candidatos = candidatos.sort_values('fec_operacion')

            pago_row = candidatos.iloc[0]
            pago_encontrado = {
                'importe': float(pago_row.get('importe', None)) if pd.notna(pago_row.get('importe', None)) else None,
                'fec_operacion': pago_row.get('fec_operacion', None),
                'cif': pago_row.get('cif', None),
                'posible_factura': pago_row.get('posible_factura', None),
            }

    # Asignar el único pago (si existe) a todas las internas seleccionadas
    if pago_encontrado:
        # crear columnas si no existen
        for c in ['Pago1_Importe', 'Pago1_Fecha']:
            if c not in df_resultado.columns:
                df_resultado[c] = pd.NA

        df_resultado.loc[:, 'Pago1_Importe'] = pago_encontrado.get('importe')
        df_resultado.loc[:, 'Pago1_Fecha'] = pago_encontrado.get('fec_operacion')
        df_resultado.loc[:, 'posible_pago'] = 'Sí'

        fstr = ''
        if isinstance(pago_encontrado.get('fec_operacion'), pd.Timestamp):
            fstr = pago_encontrado['fec_operacion'].date().isoformat()
        df_resultado.loc[:, 'pagos_detalle'] = (
            f"Pago1: {pago_encontrado.get('importe', 0):.2f} €"
            + (f" ({fstr})" if fstr else "")
            + (f" CIF: {pago_encontrado.get('cif')}" if pago_encontrado.get('cif') else "")
            + (f" Ref: {pago_encontrado.get('posible_factura')}" if pago_encontrado.get('posible_factura') else "")
        )
        st.success("✅ Pago único asignado por importe.")
    else:
        st.info("ℹ️ No se asignó pago (no se encontró uno que cumpla los criterios).")

    # --- Mostrar tabla final ---
    columnas_base = ['factura', 't_doc_n_m_doc', 'nombre_cliente', 'importe_correcto',
                     'fecha_emision', 'sociedad', 'posible_pago', 'pagos_detalle']
    columnas_base = [c for c in columnas_base if c in df_resultado.columns]
    columnas_pago = [c for c in df_resultado.columns if c.lower().startswith('pago')]
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
