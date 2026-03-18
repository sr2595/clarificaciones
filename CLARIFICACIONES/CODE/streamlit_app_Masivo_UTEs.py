import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO
import unicodedata, re, io, os, time

st.set_page_config(page_title="Clarificador UTE", page_icon="📄", layout="wide")
st.title("📄 Clarificador UTE Masivo")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _norm(texto):
    if texto is None: return ""
    s = str(texto).replace("\u00A0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

def find_col(df, candidates):
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map: return norm_map[key]
    cand_norms = [_norm(c) for c in candidates]
    for orig in df.columns:
        n = _norm(orig)
        if any(cn in n or n in cn for cn in cand_norms if cn): return orig
    return None

def conv_imp(valor):
    if pd.isna(valor): return None
    if isinstance(valor, (int, float)): return float(valor)
    try: return float(str(valor).strip().replace('.','').replace(',','.'))
    except: return None

def aplicar_impuestos(df, col_imp, col_tipo):
    factores = {"IGIC - 7":1.07,"IPSIC - 10":1.10,"IPSIM - 8":1.08,
                "IVA - 0":1.00,"IVA - 21":1.21,"EXENTO":1.0,"IVA - EXENTO":1.0}
    df[col_tipo] = df[col_tipo].astype(str).str.strip().str.upper()
    df['IMPORTE_CON_IMPUESTO'] = df.apply(
        lambda r: float(r[col_imp] * factores.get(r[col_tipo], 1.0)), axis=1)
    return df

def sociedad_por_prefijo(num):
    n = str(num).strip().upper()
    if n.startswith('60'): return 'TDE'
    if n.startswith('ADM'): return 'TME'
    return 'OTROS'

# ─────────────────────────────────────────────────────────────────────────────
# 1) PRISMA
# ─────────────────────────────────────────────────────────────────────────────
f_prisma = st.file_uploader("📂 PRISMA (CSV)", type=["csv"])
if f_prisma:
    st.session_state.prisma_bytes = f_prisma.getvalue()
    for k in ['df_prisma','df_90_base','df_90_prep']: st.session_state.pop(k, None)

if 'df_prisma' not in st.session_state:
    if 'prisma_bytes' not in st.session_state: st.stop()
    df = pd.read_csv(BytesIO(st.session_state.prisma_bytes), sep=";",
                     skiprows=1, header=0, encoding="latin1", on_bad_lines="skip")
    c_num  = find_col(df, ["Num. Factura","Factura"])
    c_cif  = find_col(df, ["CIF"])
    c_id   = find_col(df, ["id UTE"])
    c_imp  = find_col(df, ["Total Base Imponible"])
    c_tipo = find_col(df, ["Tipo Impuesto"])
    c_fec  = find_col(df, ["Fecha Emisión","Fecha"])
    faltan = [n for c,n in [(c_num,"Num.Factura"),(c_cif,"CIF"),(c_id,"Id UTE"),
                             (c_imp,"Importe"),(c_tipo,"Tipo Imp")] if not c]
    if faltan: st.error(f"❌ PRISMA: faltan columnas {faltan}"); st.stop()
    df[c_num] = df[c_num].astype(str).str.strip()
    df[c_cif] = df[c_cif].astype(str).str.replace(" ","")
    df[c_id]  = df[c_id].astype(str).str.strip()
    df['IMPORTE_CORRECTO'] = df[c_imp].apply(conv_imp)
    df[c_fec] = pd.to_datetime(df[c_fec], dayfirst=True, errors='coerce')
    df = aplicar_impuestos(df, 'IMPORTE_CORRECTO', c_tipo)
    del st.session_state.prisma_bytes
    st.session_state.df_prisma = df
    st.session_state.c_num = c_num
    st.session_state.c_cif = c_cif
    st.session_state.c_id  = c_id
    st.session_state.c_fec = c_fec
    st.success(f"✅ PRISMA: {len(df):,} filas")
else:
    df = st.session_state.df_prisma
    c_num = st.session_state.c_num
    c_cif = st.session_state.c_cif
    c_id  = st.session_state.c_id
    c_fec = st.session_state.c_fec
    st.success(f"✅ PRISMA cargado ({len(df):,} filas)")

# ─────────────────────────────────────────────────────────────────────────────
# 2) MAESTRO UTEs
# ─────────────────────────────────────────────────────────────────────────────
f_maestro = st.file_uploader("📂 Maestro UTEs (.xlsx)", type=["xlsx"], key="maestro")
if f_maestro:
    st.session_state.maestro_bytes = f_maestro.getvalue()
    st.session_state.pop('maestro_map', None)

if 'maestro_map' not in st.session_state:
    if 'maestro_bytes' in st.session_state:
        try:
            dm = pd.read_excel(BytesIO(st.session_state.maestro_bytes),
                               sheet_name="Datos", engine="openpyxl")
        except Exception as e:
            st.error(f"❌ Maestro: {e}"); st.stop()
        c_ute  = find_col(dm, ['UTE','CIF UTE','CIF_UTE'])
        c_tde  = find_col(dm, ['Porc. TdE','Porc TdE','TDE'])
        c_tme  = find_col(dm, ['Porc. TME','Porc TME','TME'])
        c_tsol = find_col(dm, ['Porc. TSOL','Porc TSOL','TSOL'])
        c_otr  = find_col(dm, ['Porc. Otros','Porc Otros','Otros'])
        if not all([c_ute, c_tde, c_tme]):
            st.error("❌ Maestro: faltan columnas UTE/TDE/TME"); st.stop()
        dm[c_ute] = dm[c_ute].astype(str).str.strip().str.upper()
        for c in [c for c in [c_tde,c_tme,c_tsol,c_otr] if c]:
            dm[c] = pd.to_numeric(dm[c].astype(str).str.replace(',','.'), errors='coerce').fillna(0.0)
        maestro_map = {}
        for _, r in dm.iterrows():
            maestro_map[str(r[c_ute])] = {
                'TDE':   float(r[c_tde]),
                'TME':   float(r[c_tme]),
                'TSOL':  float(r[c_tsol]) if c_tsol else 0.0,
                'OTROS': float(r[c_otr])  if c_otr  else 0.0,
            }
        del st.session_state.maestro_bytes
        st.session_state.maestro_map = maestro_map
        st.success(f"✅ Maestro: {len(maestro_map):,} UTEs")
else:
    st.success(f"✅ Maestro cargado ({len(st.session_state.maestro_map):,} UTEs)")

# ─────────────────────────────────────────────────────────────────────────────
# 3) PAGOS
# ─────────────────────────────────────────────────────────────────────────────
f_pagos = st.file_uploader("📂 Pagos (Cruce_Movs)", type=['xlsm','xlsx','csv'], key="cobros")
if f_pagos:
    st.session_state.cobros_bytes = f_pagos.getvalue()
    st.session_state.pop('df_cobros', None)

if 'df_cobros' not in st.session_state:
    if 'cobros_bytes' not in st.session_state: st.stop()
    data = BytesIO(st.session_state.cobros_bytes)
    xls  = pd.ExcelFile(data, engine="openpyxl")
    sheet = "Cruce_Movs" if "Cruce_Movs" in xls.sheet_names else xls.sheet_names[0]
    data.seek(0)
    dc = pd.read_excel(data, sheet_name=sheet, engine="openpyxl")
    dc.columns = (dc.columns.astype(str).str.strip().str.lower()
                  .str.replace(r'[áàäâ]','a',regex=True).str.replace(r'[éèëê]','e',regex=True)
                  .str.replace(r'[íìïî]','i',regex=True).str.replace(r'[óòöô]','o',regex=True)
                  .str.replace(r'[úùüû]','u',regex=True)
                  .str.replace(r'[^0-9a-z]','_',regex=True).str.replace(r'__+','_',regex=True).str.strip('_'))
    col_map = {'fec_operacion':['fec_operacion','fecha_operacion'],
               'importe':['importe','imp','valor'],
               'posible_factura':['posible_factura','posiblefactura'],
               'CIF_UTE':['cif','cif_ute'],
               'denominacion':['denominacion','nombre','razon_social']}
    for tgt, opts in col_map.items():
        for p in opts:
            if p in dc.columns: dc.rename(columns={p:tgt}, inplace=True); break
    if 'fec_operacion' in dc.columns:
        dc['fec_operacion'] = pd.to_datetime(dc['fec_operacion'], errors='coerce')
    if 'importe' in dc.columns:
        dc['importe'] = pd.to_numeric(dc['importe'], errors='coerce')
    if 'posible_factura' in dc.columns:
        dc['posible_factura'] = dc['posible_factura'].astype(str).str.strip()
    if 'CIF_UTE' in dc.columns:
        dc['CIF_UTE'] = dc['CIF_UTE'].astype(str).str.strip()
    del st.session_state.cobros_bytes
    st.session_state.df_cobros = dc
    st.success(f"✅ Pagos: {len(dc):,} filas")
else:
    dc = st.session_state.df_cobros
    st.success(f"✅ Pagos cargados ({len(dc):,} filas)")

# ─────────────────────────────────────────────────────────────────────────────
# 4) SELECTOR DE DÍA
# ─────────────────────────────────────────────────────────────────────────────
if 'df_cobros' not in st.session_state: st.stop()
dc = st.session_state.df_cobros
dc['fec_operacion'] = dc['fec_operacion'].dt.normalize()
dias = sorted(dc['fec_operacion'].dropna().unique())
fecha_sel = st.selectbox("📅 Día de cruce:", dias)

cols_cruce = ['fec_operacion','importe','CIF_UTE']
for c in ['posible_factura','denominacion']:
    if c in dc.columns: cols_cruce.append(c)
df_pagos = dc[dc['fec_operacion'] == pd.to_datetime(fecha_sel)][cols_cruce].copy()
st.write(f"Pagos del día: **{len(df_pagos)}** | Total: **{df_pagos['importe'].sum():,.2f} €**"
         .replace(",","X").replace(".",",").replace("X","."))

# ─────────────────────────────────────────────────────────────────────────────
# 5) PREPARAR 90s (una sola vez)
# ─────────────────────────────────────────────────────────────────────────────
if 'df_90_prep' not in st.session_state:
    df_tmp = st.session_state.df_prisma.copy()
    df_tmp[c_num] = df_tmp[c_num].astype(str).str.strip()
    df_tmp[c_id]  = df_tmp[c_id].astype(str).str.strip()
    df_tmp[c_cif] = df_tmp[c_cif].astype(str).str.strip()

    # CIF real del socio por Id UTE (primera factura no-90)
    cif_por_ute = (df_tmp[~df_tmp[c_num].str.startswith("90")]
                   .groupby(c_id)[c_cif].first().to_dict())

    df90 = df_tmp[df_tmp[c_num].str.startswith("90")].copy()
    df90['CIF_UTE_REAL'] = df90[c_id].map(lambda x: cif_por_ute.get(x, "NONE"))
    df90['Num_Factura_Norm'] = df90[c_num].str.upper()
    df90['Fecha Emisión'] = df90[c_fec]
    df90 = df90[df90['IMPORTE_CON_IMPUESTO'] > 0].copy()

    st.session_state.df_90_prep = df90
    st.success(f"✅ Facturas 90 listas: {len(df90):,}")
else:
    st.success(f"✅ Facturas 90 ya preparadas ({len(st.session_state.df_90_prep):,})")

df90 = st.session_state.df_90_prep

# ─────────────────────────────────────────────────────────────────────────────
# 6) TOLERANCIA Y BOTÓN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
tol_cent = st.number_input("Tolerancia (céntimos)", min_value=0, max_value=10000, value=0, step=1)
tolerancia = tol_cent / 100.0

col1, col2 = st.columns([3,1])
with col1: st.info(f"Día: {fecha_sel.strftime('%d/%m/%Y')} — {len(df_pagos)} pagos")
with col2: ejecutar = st.button("🔄 Ejecutar Cruce", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7) CRUCE
# ─────────────────────────────────────────────────────────────────────────────
if ejecutar:
    maestro_map = st.session_state.get('maestro_map', {})

    # Construir socios por Id UTE
    df_soc_raw = st.session_state.df_prisma.copy()
    df_soc_raw = df_soc_raw[
        (~df_soc_raw[c_num].astype(str).str.startswith("90")) &
        (df_soc_raw['IMPORTE_CON_IMPUESTO'] > 0)
    ].copy()
    df_soc_raw['SOCIEDAD'] = df_soc_raw[c_num].apply(sociedad_por_prefijo)
    socios_por_ute = {}
    for id_ute, g in df_soc_raw.groupby(c_id):
        socios_por_ute[str(id_ute).strip()] = g[[c_num,'IMPORTE_CON_IMPUESTO',c_cif,'SOCIEDAD']].copy()
    del df_soc_raw

    # Índice 90s por CIF (U y J)
    facturas_por_cif = {}
    for cif, g in df90.groupby('CIF_UTE_REAL'):
        facturas_por_cif[cif] = g
        if str(cif).startswith('U'): facturas_por_cif['J'+str(cif)[1:]] = g
        elif str(cif).startswith('J'): facturas_por_cif['U'+str(cif)[1:]] = g

    # Lookup por número de factura (posible_factura)
    num_a_fila = {str(r['Num_Factura_Norm']).strip().upper(): r
                  for _, r in df90.iterrows()}

    resultados = []
    df_pag_norm = df_pagos.copy()
    df_pag_norm['CIF_UTE'] = df_pag_norm['CIF_UTE'].astype(str).str.replace(".0","",regex=False).str.strip().str.upper()

    with st.spinner("⏳ Cruzando..."):
        inicio = time.time()
        for _, pago in df_pag_norm.iterrows():
            try:
                cif      = pago['CIF_UTE']
                imp_pago = pago['importe']
                fec_pago = pago['fec_operacion']
                cif_u    = ('U'+cif[1:]) if cif.startswith('J') else cif
                porc     = maestro_map.get(cif) or maestro_map.get(cif_u) or {}

                # posible_factura
                pos_num = str(pago.get('posible_factura','')).strip().upper()
                forzar  = num_a_fila.get(pos_num) if pos_num not in ('','NAN','NONE','N/A') else None

                # candidatas
                TOLS = max(tolerancia, 2.0)
                if forzar is not None:
                    row_d = forzar if isinstance(forzar, dict) else forzar.to_dict()
                    df_cands = pd.DataFrame([row_d])
                    for col_r, v_d in [('IMPORTE_CON_IMPUESTO',0.0),('Fecha Emisión',pd.NaT),
                                        (c_id,'DESCONOCIDO'),('Num_Factura_Norm',pos_num)]:
                        if col_r not in df_cands.columns: df_cands[col_r] = v_d
                    if str(df_cands[c_id].iloc[0]).strip() in ('','DESCONOCIDO','nan'):
                        df_cands[c_id] = str(forzar.get(c_id,'DESCONOCIDO') if isinstance(forzar,dict) else forzar.get(c_id,'DESCONOCIDO'))
                elif cif not in facturas_por_cif:
                    resultados.append({'CIF_UTE':cif,'fecha_pago':fec_pago,'importe_pago':imp_pago,
                        'facturas_90_asignadas':'SIN_90s_PARA_ESTE_CIF','importe_facturas_90':0.0,
                        'desglose_facturas_90':None,'diferencia_pago_vs_90':imp_pago,
                        'advertencia':f'Sin 90s en PRISMA para {cif}'}); continue
                else:
                    df_todas = facturas_por_cif[cif]
                    df_cands = df_todas[
                        (df_todas['IMPORTE_CON_IMPUESTO'] > 0) &
                        (df_todas['IMPORTE_CON_IMPUESTO'] <= imp_pago + TOLS) &
                        (df_todas['Fecha Emisión'].isna() | (df_todas['Fecha Emisión'] <= fec_pago))
                    ].copy()

                    # Excluir 90s con socios prohibidos por maestro (porc == 0)
                    socs_proh = {s for s,p in porc.items() if p == 0}
                    if socs_proh and not df_cands.empty:
                        validas = []
                        for _, f90r in df_cands.iterrows():
                            id90 = str(f90r.get(c_id,'DESCONOCIDO')).strip()
                            if id90 in socios_por_ute:
                                socs = set(socios_por_ute[id90]['SOCIEDAD'].unique())
                                if socs & socs_proh: continue
                            validas.append(f90r)
                        df_cands = pd.DataFrame(validas) if validas else pd.DataFrame()

                    if df_cands.empty:
                        razones = []
                        for _, f90r in df_todas.iterrows():
                            imp = f90r['IMPORTE_CON_IMPUESTO']
                            fec = f90r.get('Fecha Emisión', pd.NaT)
                            num = f90r['Num_Factura_Norm']
                            if imp > 0 and imp <= imp_pago + TOLS:
                                razones.append(f"{num} ({imp:.2f}€): fecha posterior al pago")
                            else:
                                razones.append(f"{num} ({imp:.2f}€): importe no encaja (dif={round(imp-imp_pago,2):+.2f}€)")
                        resultados.append({'CIF_UTE':cif,'fecha_pago':fec_pago,'importe_pago':imp_pago,
                            'facturas_90_asignadas':'SIN_COMBINACION_VALIDA','importe_facturas_90':0.0,
                            'desglose_facturas_90':None,'diferencia_pago_vs_90':imp_pago,
                            'advertencia':(' || '.join(razones) if razones else 'Sin candidatas')}); continue

                # Solver: preferir 1 factura, luego minimizar cantidad
                df_cands = df_cands.sort_values(['Fecha Emisión','IMPORTE_CON_IMPUESTO'])
                nums  = df_cands['Num_Factura_Norm'].tolist()
                imps  = df_cands['IMPORTE_CON_IMPUESTO'].tolist()
                iduts = df_cands[c_id].tolist()
                n     = len(imps)
                pc    = int(round(imp_pago*100))
                fc    = [int(round(x*100)) for x in imps]
                tc    = int(round(tolerancia*100))

                sel = None
                for i in range(n):
                    if abs(fc[i]-pc) <= tc: sel=[i]; break

                if sel is None:
                    m = cp_model.CpModel()
                    x = [m.NewBoolVar(f"x{i}") for i in range(n)]
                    m.Add(sum(x[i]*fc[i] for i in range(n)) >= pc-tc)
                    m.Add(sum(x[i]*fc[i] for i in range(n)) <= pc+tc)
                    m.Minimize(sum(x))
                    sv = cp_model.CpSolver()
                    sv.parameters.max_time_in_seconds = 3
                    sv.parameters.log_search_progress = False
                    st2 = sv.Solve(m)
                    if st2 not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                        resultados.append({'CIF_UTE':cif,'fecha_pago':fec_pago,'importe_pago':imp_pago,
                            'facturas_90_asignadas':'SIN_COMBINACION_EXACTA','importe_facturas_90':0.0,
                            'desglose_facturas_90':None,'diferencia_pago_vs_90':imp_pago,
                            'advertencia':f'No cuadra. 90s: {" | ".join(f"{nums[i]} ({imps[i]:.2f}€)" for i in range(n))}'}); continue
                    sel = [i for i in range(n) if sv.Value(x[i])==1]

                # Desglose socios
                desglose = []
                imp_total_90 = 0.0
                for i in sel:
                    num90  = nums[i]
                    imp90  = imps[i]
                    id_ute = str(iduts[i]).strip()
                    imp_total_90 += imp90

                    socios_p = []
                    imp_soc_p = 0.0
                    if id_ute in socios_por_ute:
                        for _, s in socios_por_ute[id_ute].iterrows():
                            socios_p.append({'num_factura':str(s[c_num]),'cif':str(s[c_cif]),
                                             'importe':float(s['IMPORTE_CON_IMPUESTO']),
                                             'fuente':f'PRISMA ({s["SOCIEDAD"]})'})
                            imp_soc_p += float(s['IMPORTE_CON_IMPUESTO'])

                    # Diferencia → estimar TSOL/OTROS por maestro
                    dif = round(imp90 - imp_soc_p, 2)
                    estimados = []
                    if abs(dif) > tolerancia and porc:
                        socs_en_prisma = {s['fuente'].split('(')[-1].rstrip(')') for s in socios_p}
                        for snom, pct in porc.items():
                            if snom in socs_en_prisma: continue
                            if pct and pct > 0:
                                estimados.append({'num_factura':'PENDIENTE','cif':snom,
                                    'importe':round(imp90*pct/100.0,2),
                                    'fuente':f'ESTIMADO ({snom} {pct:.1f}%)'})

                    dif_final = round(imp90 - imp_soc_p - sum(s['importe'] for s in estimados), 2)
                    if abs(dif_final) > tolerancia:
                        estado = f"⚠️ Diferencia sin cubrir: {dif_final:.2f}€"
                    elif estimados:
                        estado = f"✅ PRISMA + Maestro ({', '.join(s['cif'] for s in estimados)})"
                    else:
                        estado = "✅ Cuadra con socios PRISMA"

                    desglose.append({
                        'factura_90':num90,'importe_90':imp90,'caso':'PRISMA',
                        'socios':socios_p+estimados,'importe_socios':imp_soc_p+sum(s['importe'] for s in estimados),
                        'diferencia_90_socios':dif_final,
                        'socios_prisma':socios_p,'socios_estimados':estimados,
                        'importe_socios_prisma':imp_soc_p,'importe_estimado':sum(s['importe'] for s in estimados),
                        'estado':estado
                    })

                dif_pago = round(imp_pago - imp_total_90, 2)
                difs = [d for d in desglose if abs(d['diferencia_90_socios']) > tolerancia]
                resultados.append({
                    'CIF_UTE':cif,'fecha_pago':fec_pago,'importe_pago':imp_pago,
                    'facturas_90_asignadas':', '.join(d['factura_90'] for d in desglose),
                    'importe_facturas_90':imp_total_90,
                    'desglose_facturas_90':desglose,
                    'diferencia_pago_vs_90':dif_pago,
                    'advertencia':' | '.join(f"{d['factura_90']}: dif={d['diferencia_90_socios']:.2f}€" for d in difs) or None
                })
            except Exception as e:
                resultados.append({'CIF_UTE':pago.get('CIF_UTE','ERR'),'fecha_pago':pago.get('fec_operacion'),
                    'importe_pago':pago.get('importe',0),'facturas_90_asignadas':f'ERROR:{e}',
                    'importe_facturas_90':0.0,'desglose_facturas_90':None,
                    'diferencia_pago_vs_90':pago.get('importe',0),'advertencia':None})

        fin = time.time()
        df_res = pd.DataFrame(resultados)
        st.session_state.df_resultados  = df_res
        st.session_state.fecha_resultados = fecha_sel
        st.session_state.df_pag_norm    = df_pag_norm
        st.success(f"✅ Cruce completado en {fin-inicio:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 8) RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
if 'df_resultados' in st.session_state:
    df_res = st.session_state.df_resultados
    st.markdown("---")
    st.subheader("📊 Resultados")

    total = len(df_res)
    con_90 = df_res['facturas_90_asignadas'].notna().sum()
    con_adv = df_res['advertencia'].notna().sum()
    imp_pagos = df_res['importe_pago'].sum()
    imp_90s   = df_res['importe_facturas_90'].sum()

    imp_soc_p = imp_soc_e = dif_total = 0.0
    for _, row in df_res.iterrows():
        if row['desglose_facturas_90']:
            for d in row['desglose_facturas_90']:
                imp_soc_p += d['importe_socios_prisma']
                imp_soc_e += d['importe_estimado']
                dif_total += d['diferencia_90_socios']

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total pagos", total)
    c2.metric("Con 90 asignada", con_90, f"{con_90/total*100:.1f}%" if total else "")
    c3.metric("Sin 90", total-con_90)
    c4.metric("⚠️ Con advertencia", con_adv)

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    fmt = lambda x: f"{x:,.2f} €".replace(",","X").replace(".",",").replace("X",".")
    c1.metric("💰 Total pagos",   fmt(imp_pagos))
    c2.metric("🔵 Total 90s",     fmt(imp_90s))
    c3.metric("🟢 Socios PRISMA", fmt(imp_soc_p))
    c4.metric("🟡 Estimado Maestro", fmt(imp_soc_e))

    st.dataframe(df_res[['CIF_UTE','fecha_pago','importe_pago','facturas_90_asignadas',
                          'importe_facturas_90','diferencia_pago_vs_90','advertencia']],
                 use_container_width=True, height=350)

    # Excel
    cif_nombre = {}
    if 'df_pag_norm' in st.session_state and 'denominacion' in st.session_state.df_pag_norm.columns:
        for _, p in st.session_state.df_pag_norm.iterrows():
            if pd.notna(p.get('denominacion')): cif_nombre[p['CIF_UTE']] = str(p['denominacion'])

    filas = []
    for _, row in df_res.iterrows():
        nombre = cif_nombre.get(row['CIF_UTE'], '')
        if row['desglose_facturas_90']:
            for d in row['desglose_facturas_90']:
                sp_str = ' | '.join(f"{s['num_factura']} ({s['cif']}): {s['importe']:.2f}€"
                                    for s in d['socios_prisma']) or 'Sin socios PRISMA'
                se_str = ' | '.join(f"{s['cif']} {s['importe']:.2f}€ ({s['fuente']})"
                                    for s in d['socios_estimados'])
                filas.append({
                    'CIF_UTE': row['CIF_UTE'], 'Nombre_UTE': nombre,
                    'Fecha_Pago': row['fecha_pago'].date() if pd.notna(row['fecha_pago']) else None,
                    'Importe_Pago': row['importe_pago'],
                    'Factura_90': d['factura_90'], 'Importe_90': d['importe_90'],
                    'Socios_PRISMA': sp_str, 'Importe_Socios_PRISMA': d['importe_socios_prisma'],
                    'Socios_Estimados_Maestro': se_str, 'Importe_Estimado': d['importe_estimado'],
                    'Diferencia_90_vs_Socios': d['diferencia_90_socios'],
                    'Diferencia_Pago_vs_90': row['diferencia_pago_vs_90'],
                    'Estado': d['estado'],
                    'Advertencia': row['advertencia'] or ''
                })
        else:
            filas.append({
                'CIF_UTE': row['CIF_UTE'], 'Nombre_UTE': nombre,
                'Fecha_Pago': row['fecha_pago'].date() if pd.notna(row['fecha_pago']) else None,
                'Importe_Pago': row['importe_pago'],
                'Factura_90': None, 'Importe_90': 0.0,
                'Socios_PRISMA': '', 'Importe_Socios_PRISMA': 0.0,
                'Socios_Estimados_Maestro': '', 'Importe_Estimado': 0.0,
                'Diferencia_90_vs_Socios': 0.0,
                'Diferencia_Pago_vs_90': row['diferencia_pago_vs_90'],
                'Estado': row['facturas_90_asignadas'] or '',
                'Advertencia': row['advertencia'] or ''
            })

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        pd.DataFrame(filas).to_excel(w, index=False, sheet_name="Resultados")
    out.seek(0)
    st.download_button("📥 Descargar Excel", data=out,
        file_name=f"cruce_{st.session_state.fecha_resultados.strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
