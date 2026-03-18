import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO
from datetime import datetime
import unicodedata, re
import io
import os
import time

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

def aplicar_impuestos_a_prisma(df_prisma, col_importe='IMPORTE_CORRECTO', col_tipo_impuesto='Tipo Impuesto'):
    factores = {
        "IGIC - 7": 1.07, "IPSIC - 10": 1.10, "IPSIM - 8": 1.08,
        "IVA - 0": 1.00, "IVA - 21": 1.21, "EXENTO": 1.0, "IVA - EXENTO": 1.0,
    }
    df_prisma[col_tipo_impuesto] = df_prisma[col_tipo_impuesto].astype(str).str.strip().str.upper()
    df_prisma['IMPORTE_CON_IMPUESTO'] = df_prisma.apply(
        lambda row: float(row[col_importe] * factores.get(row[col_tipo_impuesto], 1.0)), axis=1)
    return df_prisma

def sociedad_por_prefijo(num):
    n = str(num).strip().upper()
    if n.startswith('60'): return 'TDE'
    if n.startswith('ADM'): return 'TME'
    return 'OTROS'

# --------- 1) PRISMA ---------
archivo_prisma = st.file_uploader("Sube el archivo PRISMA (CSV)", type=["csv"])

if archivo_prisma is not None:
    if st.session_state.get('prisma_file_id') != archivo_prisma.file_id:
        st.session_state.prisma_file_id = archivo_prisma.file_id
        st.session_state.prisma_bytes   = archivo_prisma.getvalue()
        for k in ['df_prisma_procesado','df_prisma_90_base','df_prisma_90_preparado']:
            st.session_state.pop(k, None)

if "prisma_bytes" not in st.session_state and "df_prisma_procesado" not in st.session_state:
    st.stop()

if "df_prisma_procesado" not in st.session_state:
    st.info("⏳ Procesando archivo PRISMA...")
    df_prisma = pd.read_csv(BytesIO(st.session_state.prisma_bytes), sep=";",
                            skiprows=1, header=0, encoding="latin1", on_bad_lines="skip")
    col_id_ute_prisma      = find_col(df_prisma, ["id UTE"])
    col_num_factura_prisma = find_col(df_prisma, ["Num. Factura", "Factura"])
    col_fecha_prisma       = find_col(df_prisma, ["Fecha Emisión", "Fecha"])
    col_cif_prisma         = find_col(df_prisma, ["CIF"])
    col_importe_prisma     = find_col(df_prisma, ["Total Base Imponible"])
    col_tipo_imp_prisma    = find_col(df_prisma, ["Tipo Impuesto"])
    faltan = [n for c,n in [(col_id_ute_prisma,"id UTE"),(col_num_factura_prisma,"Num. Factura"),
                             (col_cif_prisma,"CIF"),(col_importe_prisma,"Total Base Imponible"),
                             (col_tipo_imp_prisma,"Tipo Impuesto")] if not c]
    if faltan: st.error(f"❌ PRISMA: faltan columnas {faltan}"); st.stop()
    df_prisma[col_num_factura_prisma] = df_prisma[col_num_factura_prisma].astype(str).str.strip()
    df_prisma[col_cif_prisma]         = df_prisma[col_cif_prisma].astype(str).str.replace(" ","")
    df_prisma[col_id_ute_prisma]      = df_prisma[col_id_ute_prisma].astype(str).str.strip()
    df_prisma['IMPORTE_CORRECTO']     = df_prisma[col_importe_prisma].apply(convertir_importe_europeo)
    df_prisma[col_fecha_prisma]       = pd.to_datetime(df_prisma[col_fecha_prisma], dayfirst=True, errors='coerce')
    df_prisma = aplicar_impuestos_a_prisma(df_prisma, col_tipo_impuesto=col_tipo_imp_prisma)
    del st.session_state.prisma_bytes
    st.session_state.df_prisma_procesado    = df_prisma
    st.session_state.col_num_factura_prisma = col_num_factura_prisma
    st.session_state.col_cif_prisma         = col_cif_prisma
    st.session_state.col_id_ute_prisma      = col_id_ute_prisma
    st.session_state.col_fecha_prisma       = col_fecha_prisma
    st.success(f"✅ PRISMA cargado: {len(df_prisma):,} filas")
else:
    df_prisma              = st.session_state.df_prisma_procesado
    col_num_factura_prisma = st.session_state.col_num_factura_prisma
    col_cif_prisma         = st.session_state.col_cif_prisma
    col_id_ute_prisma      = st.session_state.col_id_ute_prisma
    col_fecha_prisma       = st.session_state.col_fecha_prisma
    st.success(f"✅ PRISMA cargado ({len(df_prisma):,} filas)")

# --------- 2) MAESTRO UTEs ---------
archivo_maestro = st.file_uploader("Sube el Maestro UTEs (.xlsx)", type=["xlsx"], key="maestro")
if archivo_maestro is not None:
    if st.session_state.get('maestro_file_id') != archivo_maestro.file_id:
        st.session_state.maestro_file_id = archivo_maestro.file_id
        st.session_state.maestro_bytes   = archivo_maestro.getvalue()
        st.session_state.pop('maestro_map', None)
        st.session_state.pop('df_maestro_utes', None)

if "df_maestro_utes" not in st.session_state and "maestro_bytes" in st.session_state:
    st.info("⏳ Procesando Maestro UTEs...")
    try:
        df_maestro = pd.read_excel(BytesIO(st.session_state.maestro_bytes), sheet_name="Datos", engine="openpyxl")
    except Exception as e:
        st.error(f"❌ Error leyendo Maestro UTEs: {e}"); st.stop()
    c_ute  = find_col(df_maestro, ['UTE','CIF UTE','CIF_UTE'])
    c_tde  = find_col(df_maestro, ['Porc. TdE','Porc TdE','TDE'])
    c_tme  = find_col(df_maestro, ['Porc. TME','Porc TME','TME'])
    c_tsol = find_col(df_maestro, ['Porc. TSOL','Porc TSOL','TSOL'])
    c_otr  = find_col(df_maestro, ['Porc. Otros','Porc Otros','Otros'])
    if not all([c_ute,c_tde,c_tme]):
        st.error("❌ Maestro UTEs: faltan columnas UTE/TDE/TME"); st.stop()
    df_maestro[c_ute] = df_maestro[c_ute].astype(str).str.strip().str.upper()
    for c in [c for c in [c_tde,c_tme,c_tsol,c_otr] if c]:
        df_maestro[c] = pd.to_numeric(df_maestro[c].astype(str).str.replace(',','.'), errors='coerce').fillna(0.0)
    maestro_map = {}
    for _, row in df_maestro.iterrows():
        maestro_map[str(row[c_ute])] = {
            'TDE':   float(row[c_tde]),
            'TME':   float(row[c_tme]),
            'TSOL':  float(row[c_tsol]) if c_tsol else 0.0,
            'OTROS': float(row[c_otr])  if c_otr  else 0.0,
        }
    del st.session_state.maestro_bytes
    st.session_state.df_maestro_utes = df_maestro
    st.session_state.maestro_map     = maestro_map
    st.success(f"✅ Maestro UTEs cargado: {len(maestro_map):,} UTEs")
elif "df_maestro_utes" in st.session_state:
    st.success(f"✅ Maestro UTEs cargado ({len(st.session_state.maestro_map):,} UTEs)")

# --------- 3) PAGOS ---------
cobros_file = st.file_uploader("Sube el Excel de pagos (Cruce_Movs)", type=['xlsm','xlsx','csv'], key="cobros")
if cobros_file is not None:
    if st.session_state.get('cobros_file_id') != cobros_file.file_id:
        st.session_state.cobros_file_id = cobros_file.file_id
        st.session_state.cobros_bytes   = cobros_file.getvalue()
        st.session_state.pop('df_cobros_procesado', None)

if "cobros_bytes" not in st.session_state and "df_cobros_procesado" not in st.session_state:
    st.stop()

if "df_cobros_procesado" not in st.session_state:
    st.info("⏳ Procesando archivo de PAGOS...")
    data = BytesIO(st.session_state.cobros_bytes)
    xls  = pd.ExcelFile(data, engine="openpyxl")
    sheet = "Cruce_Movs" if "Cruce_Movs" in xls.sheet_names else xls.sheet_names[0]
    data.seek(0)
    df_cobros = pd.read_excel(data, sheet_name=sheet, engine="openpyxl")
    df_cobros.columns = (df_cobros.columns.astype(str).str.strip().str.lower()
        .str.replace(r'[áàäâ]','a',regex=True).str.replace(r'[éèëê]','e',regex=True)
        .str.replace(r'[íìïî]','i',regex=True).str.replace(r'[óòöô]','o',regex=True)
        .str.replace(r'[úùüû]','u',regex=True)
        .str.replace(r'[^0-9a-z]','_',regex=True).str.replace(r'__+','_',regex=True).str.strip('_'))
    col_map = {
        'fec_operacion': ['fec_operacion','fecha_operacion','fec_oper'],
        'importe':       ['importe','imp','monto','amount','valor'],
        'posible_factura':['posible_factura','factura','posiblefactura'],
        'CIF_UTE':       ['cif','cif_ute'],
        'denominacion':  ['denominacion','nombre','razon_social','nombre_ute']
    }
    for tgt, opts in col_map.items():
        for p in opts:
            if p in df_cobros.columns: df_cobros.rename(columns={p:tgt}, inplace=True); break
    if 'fec_operacion'   in df_cobros.columns: df_cobros['fec_operacion']   = pd.to_datetime(df_cobros['fec_operacion'], errors='coerce')
    if 'importe'         in df_cobros.columns: df_cobros['importe']         = pd.to_numeric(df_cobros['importe'], errors='coerce')
    if 'posible_factura' in df_cobros.columns: df_cobros['posible_factura'] = df_cobros['posible_factura'].astype(str).str.strip()
    if 'CIF_UTE'         in df_cobros.columns: df_cobros['CIF_UTE']         = df_cobros['CIF_UTE'].astype(str).str.strip()
    del st.session_state.cobros_bytes
    st.session_state.df_cobros_procesado = df_cobros
    st.success(f"✅ Pagos cargados: {len(df_cobros):,} filas")
else:
    df_cobros = st.session_state.df_cobros_procesado
    st.success(f"✅ Pagos cargados ({len(df_cobros):,} filas)")

# --------- 4) SELECTOR DÍA ---------
if not df_cobros.empty:
    df_cobros['fec_operacion'] = df_cobros['fec_operacion'].dt.normalize()
    dias_disponibles = sorted(df_cobros['fec_operacion'].dropna().unique())
    fecha_seleccionada = st.selectbox("📅 Selecciona el día:", dias_disponibles)

    df_cobros_filtrado = df_cobros[
        df_cobros['fec_operacion'].notna() &
        (df_cobros['fec_operacion'].dt.normalize() == pd.to_datetime(fecha_seleccionada))
    ].copy()
    st.write(f"ℹ️ Pagos del día: **{len(df_cobros_filtrado)}** | Total: **{df_cobros_filtrado['importe'].sum():,.2f} €**"
             .replace(",","X").replace(".",",").replace("X","."))

    columnas_cruce = ['fec_operacion','importe','posible_factura','CIF_UTE']
    if 'denominacion' in df_cobros_filtrado.columns: columnas_cruce.append('denominacion')
    df_pagos = df_cobros_filtrado[columnas_cruce].copy()

    # --------- 5) PREPARAR 90s (una sola vez) ---------
    if "df_prisma_90_base" not in st.session_state:
        df_prisma[col_cif_prisma] = df_prisma[col_cif_prisma].astype(str).str.replace(".0","",regex=False).str.strip().str.upper()
        df_temp = df_prisma.copy()
        df_temp[col_num_factura_prisma] = df_temp[col_num_factura_prisma].astype(str).str.strip()
        df_temp[col_id_ute_prisma]      = df_temp[col_id_ute_prisma].astype(str).str.strip()
        df_temp[col_cif_prisma]         = df_temp[col_cif_prisma].astype(str).str.strip()
        cif_por_ute = df_temp[~df_temp[col_num_factura_prisma].str.startswith("90")].groupby(col_id_ute_prisma)[col_cif_prisma].first().to_dict()
        df90_base = df_temp[df_temp[col_num_factura_prisma].str.startswith("90")].copy()
        df90_base['CIF_UTE_REAL']     = df90_base[col_id_ute_prisma].apply(lambda x: cif_por_ute.get(x,"NONE"))
        df90_base['Num_Factura_Norm'] = df90_base[col_num_factura_prisma].astype(str).str.strip().str.upper()
        st.session_state.df_prisma_90_base = df90_base
        st.success(f"✅ Base PRISMA preparada: {len(df90_base):,} facturas 90")
    else:
        df90_base = st.session_state.df_prisma_90_base
        st.success(f"✅ Base PRISMA ya cargada ({len(df90_base):,} facturas 90)")

    if "df_prisma_90_preparado" not in st.session_state:
        df90 = df90_base.copy()
        df90['Fecha Emisión'] = df90[col_fecha_prisma]
        df90 = df90[df90['IMPORTE_CON_IMPUESTO'] > 0].copy()
        st.session_state.df_prisma_90_preparado = df90
        st.success(f"✅ Facturas 90 listas: {len(df90):,}")
    else:
        df90 = st.session_state.df_prisma_90_preparado
        st.success(f"✅ Facturas 90 ya cargadas ({len(df90):,})")

    # --------- 6) TOLERANCIA Y BOTÓN ---------
    st.markdown("---")
    col_tol, _ = st.columns([1,2])
    with col_tol:
        tolerancia_centimos = st.number_input("Tolerancia (céntimos)", min_value=0, max_value=10000, value=0, step=1)
    tolerancia_euros = tolerancia_centimos / 100.0

    col1, col2 = st.columns([3,1])
    with col1: st.info(f"📅 Día: **{fecha_seleccionada.strftime('%d/%m/%Y')}** ({len(df_pagos)} pagos)")
    with col2: ejecutar_cruce = st.button("🔄 Ejecutar Cruce", type="primary", use_container_width=True)

    # --------- 7) CRUCE ---------
    if ejecutar_cruce:
        with st.spinner("⏳ Buscando combinaciones óptimas de facturas..."):
            inicio = time.time()
            maestro_map = st.session_state.get('maestro_map', {})

            # Socios por Id UTE
            df_soc_raw = df_prisma.copy()
            df_soc_raw = df_soc_raw[
                (~df_soc_raw[col_num_factura_prisma].astype(str).str.startswith("90")) &
                (df_soc_raw['IMPORTE_CON_IMPUESTO'] > 0)
            ].copy()
            df_soc_raw['SOCIEDAD_PRISMA'] = df_soc_raw[col_num_factura_prisma].apply(sociedad_por_prefijo)
            socios_por_ute = {}
            for id_ute, g in df_soc_raw.groupby(col_id_ute_prisma):
                socios_por_ute[str(id_ute).strip()] = g[[col_num_factura_prisma,'IMPORTE_CON_IMPUESTO',col_cif_prisma,'SOCIEDAD_PRISMA']].copy()
            del df_soc_raw

            # Índice 90s por CIF
            facturas_por_cif = {}
            for cif, g in df90.groupby('CIF_UTE_REAL'):
                facturas_por_cif[cif] = g
                if str(cif).startswith('U'): facturas_por_cif['J'+str(cif)[1:]] = g
                elif str(cif).startswith('J'): facturas_por_cif['U'+str(cif)[1:]] = g

            todas_90_por_num = {str(r['Num_Factura_Norm']).strip().upper(): r for _, r in df90.iterrows()}

            resultados = []
            df_pagos_normalizado = df_pagos.copy()
            df_pagos_normalizado['CIF_UTE'] = df_pagos_normalizado['CIF_UTE'].astype(str).str.replace(".0","",regex=False).str.strip().str.upper()

            for _, pago in df_pagos_normalizado.iterrows():
                try:
                    cif_pago     = pago['CIF_UTE']
                    importe_pago = pago['importe']
                    fecha_pago   = pago['fec_operacion']
                    cif_u = ('U'+cif_pago[1:]) if cif_pago.startswith('J') else cif_pago
                    porcentajes = maestro_map.get(cif_pago) or maestro_map.get(cif_u) or {}

                    posible_num = str(pago.get('posible_factura','')).strip().upper()
                    forzar_90   = todas_90_por_num.get(posible_num) if posible_num not in ('','NAN','NONE','N/A') else None

                    if cif_pago not in facturas_por_cif and forzar_90 is None:
                        resultados.append({'CIF_UTE':cif_pago,'fecha_pago':fecha_pago,'importe_pago':importe_pago,
                            'facturas_90_asignadas':'SIN_90s_PARA_ESTE_CIF','importe_facturas_90':0.0,
                            'desglose_facturas_90':None,'diferencia_pago_vs_90':importe_pago,
                            'advertencia':f'Sin 90s en PRISMA para {cif_pago}'}); continue

                    TOLERANCIA_90 = max(tolerancia_euros, 2.0)

                    if forzar_90 is not None:
                        forzar_row = forzar_90 if isinstance(forzar_90, dict) else forzar_90.to_dict()
                        df_facturas = pd.DataFrame([forzar_row])
                        for col_r, v_d in [('IMPORTE_CON_IMPUESTO',0.0),('Fecha Emisión',pd.NaT),
                                            (col_id_ute_prisma,'DESCONOCIDO'),('Num_Factura_Norm',posible_num)]:
                            if col_r not in df_facturas.columns: df_facturas[col_r] = v_d
                        if str(df_facturas[col_id_ute_prisma].iloc[0]).strip() in ('','DESCONOCIDO','nan'):
                            id_ute_real = todas_90_por_num[posible_num].get(col_id_ute_prisma,'DESCONOCIDO') if isinstance(todas_90_por_num.get(posible_num),dict) else getattr(todas_90_por_num.get(posible_num),col_id_ute_prisma,'DESCONOCIDO')
                            df_facturas[col_id_ute_prisma] = str(id_ute_real).strip()
                    else:
                        df_todas = facturas_por_cif[cif_pago]
                        df_cands = df_todas[
                            (df_todas['IMPORTE_CON_IMPUESTO'] > 0) &
                            (df_todas['IMPORTE_CON_IMPUESTO'] <= importe_pago + TOLERANCIA_90) &
                            (df_todas['Fecha Emisión'].isna() | (df_todas['Fecha Emisión'] <= fecha_pago))
                        ].copy()

                        socs_proh = {s for s,p in porcentajes.items() if p == 0}
                        if socs_proh and not df_cands.empty:
                            validas = []
                            for _, f90r in df_cands.iterrows():
                                id90 = str(f90r.get(col_id_ute_prisma,'DESCONOCIDO')).strip()
                                if id90 in socios_por_ute:
                                    socs = set(socios_por_ute[id90]['SOCIEDAD_PRISMA'].unique())
                                    if socs & socs_proh: continue
                                validas.append(f90r)
                            df_cands = pd.DataFrame(validas) if validas else pd.DataFrame()

                        df_facturas = df_cands

                        if df_facturas.empty:
                            razones = []
                            for _, f90r in df_todas.iterrows():
                                imp = f90r['IMPORTE_CON_IMPUESTO']
                                fec = f90r.get('Fecha Emisión', pd.NaT)
                                num = f90r['Num_Factura_Norm']
                                if imp > 0 and imp <= importe_pago + TOLERANCIA_90:
                                    fec_str = fec.strftime('%d/%m/%Y') if pd.notna(fec) else 'sin fecha'
                                    razones.append(f"{num} ({imp:.2f}€): FECHA POSTERIOR AL PAGO ({fec_str})")
                                else:
                                    razones.append(f"{num} ({imp:.2f}€): importe no encaja (dif={round(imp-importe_pago,2):+.2f}€)")
                            resultados.append({'CIF_UTE':cif_pago,'fecha_pago':fecha_pago,'importe_pago':importe_pago,
                                'facturas_90_asignadas':'SIN_COMBINACION_VALIDA','importe_facturas_90':0.0,
                                'desglose_facturas_90':None,'diferencia_pago_vs_90':importe_pago,
                                'advertencia':' || '.join(razones) if razones else 'Sin candidatas'}); continue

                    # Solver
                    df_facturas = df_facturas.sort_values(['Fecha Emisión','IMPORTE_CON_IMPUESTO'])
                    numeros_facturas  = df_facturas['Num_Factura_Norm'].tolist()
                    importes_facturas = df_facturas['IMPORTE_CON_IMPUESTO'].tolist()
                    ids_ute           = df_facturas[col_id_ute_prisma].tolist()
                    n          = len(importes_facturas)
                    pagos_cent = int(round(importe_pago*100))
                    fact_cent  = [int(round(f*100)) for f in importes_facturas]
                    tol_cent   = int(round(tolerancia_euros*100))

                    seleccion = None
                    for i in range(n):
                        if abs(fact_cent[i]-pagos_cent) <= tol_cent: seleccion=[i]; break

                    if seleccion is None:
                        model = cp_model.CpModel()
                        x = [model.NewBoolVar(f"x{i}") for i in range(n)]
                        model.Add(sum(x[i]*fact_cent[i] for i in range(n)) >= pagos_cent-tol_cent)
                        model.Add(sum(x[i]*fact_cent[i] for i in range(n)) <= pagos_cent+tol_cent)
                        # Objetivo: minimizar nº facturas primero, y como desempate
                        # priorizar las más antiguas (índice menor = más antigua por el sort previo)
                        # Multiplicamos el índice por un peso pequeño para no interferir con el objetivo principal
                        peso_orden = [i for i in range(n)]  # 0=más antigua, n-1=más nueva
                        model.Minimize(sum(x[i] for i in range(n)) * 10000 + sum(x[i]*peso_orden[i] for i in range(n)))
                        solver = cp_model.CpSolver()
                        solver.parameters.max_time_in_seconds = 3
                        solver.parameters.log_search_progress = False
                        status = solver.Solve(model)
                        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                            resultados.append({'CIF_UTE':cif_pago,'fecha_pago':fecha_pago,'importe_pago':importe_pago,
                                'facturas_90_asignadas':'SIN_COMBINACION_EXACTA','importe_facturas_90':0.0,
                                'desglose_facturas_90':None,'diferencia_pago_vs_90':importe_pago,
                                'advertencia':f'No cuadra. 90s: {" | ".join(f"{numeros_facturas[i]} ({importes_facturas[i]:.2f}€)" for i in range(n))}'}); continue
                        seleccion = [i for i in range(n) if solver.Value(x[i])==1]

                    desglose_por_factura_90 = []
                    importe_facturas_90 = 0.0

                    for i in seleccion:
                        num_90  = numeros_facturas[i]
                        imp_90  = importes_facturas[i]
                        id_ute  = str(ids_ute[i]).strip()
                        importe_facturas_90 += imp_90

                        socios_prisma = []
                        importe_socios_prisma = 0.0
                        if id_ute in socios_por_ute:
                            for _, socio in socios_por_ute[id_ute].iterrows():
                                sociedad = str(socio.get('SOCIEDAD_PRISMA','OTROS'))
                                socios_prisma.append({'num_factura':str(socio[col_num_factura_prisma]),
                                    'cif':str(socio[col_cif_prisma]),'importe':float(socio['IMPORTE_CON_IMPUESTO']),
                                    'fuente':f'PRISMA ({sociedad})'})
                                importe_socios_prisma += float(socio['IMPORTE_CON_IMPUESTO'])

                        diferencia = round(imp_90 - importe_socios_prisma, 2)
                        socios_estimados = []
                        if abs(diferencia) > tolerancia_euros and porcentajes:
                            socs_en_prisma = {s['fuente'].split('(')[-1].rstrip(')') for s in socios_prisma}
                            for soc_nombre, porc in porcentajes.items():
                                if soc_nombre in socs_en_prisma: continue
                                if porc and porc > 0:
                                    socios_estimados.append({'num_factura':'PENDIENTE','cif':soc_nombre,
                                        'importe':round(imp_90*porc/100.0,2),
                                        'fuente':f'ESTIMADO_MAESTRO ({soc_nombre} {porc:.1f}%)'})

                        diferencia_final = round(imp_90 - importe_socios_prisma - sum(s['importe'] for s in socios_estimados), 2)
                        if abs(diferencia_final) > tolerancia_euros:
                            estado = f"⚠️ Diferencia sin cubrir: {diferencia_final:.2f}€"
                        elif socios_estimados:
                            estado = f"✅ Socios PRISMA + estimados por Maestro ({', '.join(s['cif'] for s in socios_estimados)})"
                        else:
                            estado = "✅ Cuadra con socios PRISMA"

                        desglose_por_factura_90.append({
                            'factura_90':num_90,'importe_90':imp_90,'caso':'PRISMA',
                            'socios':socios_prisma+socios_estimados,
                            'importe_socios':importe_socios_prisma+sum(s['importe'] for s in socios_estimados),
                            'diferencia_90_socios':diferencia_final,
                            'socios_prisma':socios_prisma,'socios_cobra':socios_estimados,
                            'importe_socios_prisma':importe_socios_prisma,
                            'importe_socios_cobra':sum(s['importe'] for s in socios_estimados),
                            'estado_cobra':estado
                        })

                    facturas_90_str = ', '.join([d['factura_90'] for d in desglose_por_factura_90])
                    diferencia_pago_vs_90 = round(importe_pago - importe_facturas_90, 2)
                    difs = [d for d in desglose_por_factura_90 if abs(d['diferencia_90_socios']) > tolerancia_euros]
                    advertencia = ' | '.join([f"{d['factura_90']}: dif={d['diferencia_90_socios']:.2f}€" for d in difs]) or None

                    resultados.append({'CIF_UTE':cif_pago,'fecha_pago':fecha_pago,'importe_pago':importe_pago,
                        'facturas_90_asignadas':facturas_90_str,'importe_facturas_90':importe_facturas_90,
                        'desglose_facturas_90':desglose_por_factura_90,
                        'diferencia_pago_vs_90':diferencia_pago_vs_90,'advertencia':advertencia})

                except Exception as e:
                    resultados.append({'CIF_UTE':pago.get('CIF_UTE','ERROR'),'fecha_pago':pago.get('fec_operacion'),
                        'importe_pago':pago.get('importe',0),'facturas_90_asignadas':f'ERROR:{e}',
                        'importe_facturas_90':0.0,'desglose_facturas_90':None,
                        'diferencia_pago_vs_90':pago.get('importe',0),'advertencia':None})

            fin = time.time()
            df_resultados = pd.DataFrame(resultados)
            st.session_state.df_resultados        = df_resultados
            st.session_state.fecha_resultados     = fecha_seleccionada
            st.session_state.df_pagos_normalizado = df_pagos_normalizado
            st.success(f"✅ Cruce completado en {fin-inicio:.2f} segundos")

    # --------- 8) RESULTADOS ---------
    if "df_resultados" in st.session_state and st.session_state.df_resultados is not None:
        df_resultados = st.session_state.df_resultados
        st.markdown("---")
        st.subheader("📊 Resultados del cruce")

        total_pagos           = len(df_resultados)
        pagos_con_facturas    = df_resultados['facturas_90_asignadas'].notna().sum()
        pagos_con_advertencia = df_resultados['advertencia'].notna().sum()
        importe_total_pagos   = df_resultados['importe_pago'].sum()
        importe_total_90      = df_resultados['importe_facturas_90'].sum()
        diferencia_pago_vs_90 = df_resultados['diferencia_pago_vs_90'].sum()

        importe_total_socios = diferencia_total_90_vs_socios = 0.0
        for _, row in df_resultados.iterrows():
            if row['desglose_facturas_90']:
                for f90 in row['desglose_facturas_90']:
                    importe_total_socios         += f90['importe_socios']
                    diferencia_total_90_vs_socios += f90['diferencia_90_socios']

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Pagos", total_pagos)
        c2.metric("Con Facturas 90", pagos_con_facturas, f"{pagos_con_facturas/total_pagos*100:.1f}%" if total_pagos else "")
        c3.metric("Sin Facturas", total_pagos-pagos_con_facturas)
        c4.metric("⚠️ Con Advertencia", pagos_con_advertencia)

        st.markdown("---")
        c1,c2,c3,c4 = st.columns(4)
        fmt = lambda x: f"{x:,.2f} €".replace(",","X").replace(".",",").replace("X",".")
        c1.metric("💰 Total Pagos",    fmt(importe_total_pagos))
        c2.metric("🔵 Facturas 90",    fmt(importe_total_90))
        c3.metric("🟢 Facturas Socios",fmt(importe_total_socios))
        c4.metric("⚠️ Dif. 90 vs Socios", fmt(diferencia_total_90_vs_socios))

        if pagos_con_advertencia > 0:
            st.warning(f"⚠️ {pagos_con_advertencia} pago(s) con advertencia — revisa la columna 'Advertencia' en el Excel.")

        st.dataframe(df_resultados[['CIF_UTE','fecha_pago','importe_pago','facturas_90_asignadas',
                                    'importe_facturas_90','diferencia_pago_vs_90','advertencia']],
                     use_container_width=True, height=400)

        # --------- 9) EXCEL ---------
        cif_a_nombre = {}
        if 'denominacion' in st.session_state.df_pagos_normalizado.columns:
            for _, p in st.session_state.df_pagos_normalizado.iterrows():
                if pd.notna(p.get('denominacion')): cif_a_nombre[p['CIF_UTE']] = str(p['denominacion'])

        filas_excel = []
        for _, row in df_resultados.iterrows():
            nombre_ute = cif_a_nombre.get(row['CIF_UTE'], 'DESCONOCIDO')
            if row['desglose_facturas_90']:
                for f90 in row['desglose_facturas_90']:
                    sp_str = ' | '.join(f"{s['num_factura']} ({s['cif']}): {s['importe']:.2f}€" for s in f90.get('socios_prisma',[])) or 'Sin socios en PRISMA'
                    se_str = ' | '.join(f"{s['num_factura']} ({s['fuente']}): {s['importe']:.2f}€" for s in f90.get('socios_cobra',[]))
                    filas_excel.append({
                        'CIF_UTE': row['CIF_UTE'], 'Nombre_UTE': nombre_ute,
                        'Fecha_Pago': row['fecha_pago'].date() if pd.notna(row['fecha_pago']) else None,
                        'Importe_Pago': row['importe_pago'],
                        'Factura_90': f90['factura_90'], 'Caso': f90.get('caso',''),
                        'Importe_90': f90['importe_90'],
                        'Socios_PRISMA': sp_str, 'Importe_Socios_PRISMA': f90.get('importe_socios_prisma',0),
                        'Socios_Estimados': se_str, 'Importe_Estimado': f90.get('importe_socios_cobra',0),
                        'Estado': f90.get('estado_cobra',''),
                        'Total_Socios': f90['importe_socios'],
                        'Diferencia_90_vs_Socios': f90['diferencia_90_socios'],
                        'Diferencia_Pago_vs_90': row['diferencia_pago_vs_90'],
                        'Advertencia': row['advertencia'] if pd.notna(row['advertencia']) else ''
                    })
            else:
                filas_excel.append({
                    'CIF_UTE': row['CIF_UTE'], 'Nombre_UTE': nombre_ute,
                    'Fecha_Pago': row['fecha_pago'].date() if pd.notna(row['fecha_pago']) else None,
                    'Importe_Pago': row['importe_pago'],
                    'Factura_90': None, 'Caso': '', 'Importe_90': 0.0,
                    'Socios_PRISMA': None, 'Importe_Socios_PRISMA': 0.0,
                    'Socios_Estimados': None, 'Importe_Estimado': 0.0,
                    'Estado': row['facturas_90_asignadas'] or '',
                    'Total_Socios': 0.0, 'Diferencia_90_vs_Socios': 0.0,
                    'Diferencia_Pago_vs_90': row['diferencia_pago_vs_90'],
                    'Advertencia': row['advertencia'] if pd.notna(row['advertencia']) else ''
                })

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame(filas_excel).to_excel(writer, index=False, sheet_name="Desglose_Detallado")
        output.seek(0)
        st.download_button("📥 Descargar resultados en Excel", data=output,
            file_name=f"resultados_cruce_{st.session_state.fecha_resultados.strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

    elif not ejecutar_cruce and "df_resultados" not in st.session_state:
        st.info("👆 Pulsa el botón 'Ejecutar Cruce' para iniciar el proceso")
