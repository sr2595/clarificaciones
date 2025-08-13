import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Clarificador 2.0 COBRA", page_icon="📄", layout="wide")
st.title("📄 Clarificador 2.0 COBRA")

# --- Subir archivo Excel ---
archivo = st.file_uploader("Sube el archivo Excel", type=["xlsx", "xls"])
if archivo:
    # Lee con openpyxl si está disponible; si no, deja que pandas elija motor
    try:
        df = pd.read_excel(archivo, engine="openpyxl")
    except Exception:
        df = pd.read_excel(archivo)

    # --- Detectar columnas ---
    possible_date_cols = ['FECHA', 'Fecha', 'fecha', 'Fecha Emision', 'FECHA_EMISION', 'Fecha Emisión', 'FX_EMISION']
    col_fecha_emision = next((col for col in possible_date_cols if col in df.columns), None)

    possible_factura_cols = ['FACTURA', 'Factura', 'factura', 'Nº Factura', 'NRO_FACTURA', 'Núm.Doc.Deuda']
    col_factura = next((col for col in possible_factura_cols if col in df.columns), None)

    possible_importe_cols = ['IMPORTE', 'Importe', 'importe', 'TOTAL', 'TOTAL_FACTURA']
    col_importe = next((col for col in possible_importe_cols if col in df.columns), None)

    if not (col_fecha_emision and col_factura and col_importe):
        st.error("❌ No se encontraron todas las columnas necesarias (fecha, factura, importe).")
        st.stop()

    # --- Procesar datos ---
    df[col_fecha_emision] = pd.to_datetime(df[col_fecha_emision], dayfirst=True, errors='coerce')
    df[col_factura] = df[col_factura].astype(str)

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

    df['IMPORTE_CORRECTO'] = df[col_importe].apply(convertir_importe_europeo)

    total = df['IMPORTE_CORRECTO'].sum(skipna=True)
    minimo = df['IMPORTE_CORRECTO'].min(skipna=True)
    maximo = df['IMPORTE_CORRECTO'].max(skipna=True)

    st.write("**📊 Resumen del archivo:**")
    st.write(f"- Número total de facturas: {len(df)}")
    st.write(f"- Suma total importes: {total:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe mínimo: {minimo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe máximo: {maximo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

    # --- Inputs para búsqueda ---
    importe_objetivo = st.text_input("Introduce importe objetivo (ej: 295.206,63)")
    fecha_pago = st.date_input("Fecha de pago")

    if importe_objetivo:
        # Validación e interpretación del importe
        try:
            importe_objetivo_eur = float(importe_objetivo.replace('.', '').replace(',', '.'))
            importe_objetivo_cent = int(round(importe_objetivo_eur * 100))
        except Exception:
            st.error("Formato de importe no válido.")
            st.stop()

        # Fechas de referencia para costes de desempate
        fecha_base = df[col_fecha_emision].min()
        if pd.isna(fecha_base):
            st.error("❌ La columna de fechas no contiene valores válidos.")
            st.stop()

        df['DAYS_FROM_BASE'] = (df[col_fecha_emision] - fecha_base).dt.days.fillna(0).astype(int)
        df['IMPORTE_CENT'] = (df['IMPORTE_CORRECTO'] * 100).round().astype('Int64')

        # --- Filtrar solo facturas positivas y con importe válido ---
        df_positivas = df[(df['IMPORTE_CORRECTO'] > 0) & df['IMPORTE_CENT'].notna()].copy()
        if df_positivas.empty:
            st.warning("No hay facturas positivas con importes válidos.")
            st.stop()

        # --- Función OR-Tools ---
        def seleccionar_facturas_exactas_ortools(df_filtrado, objetivo_cent, target_days):
            """
            1) Suma exacta de importes en cent.
            2) Minimiza número de facturas.
            3) Desempata favoreciendo fechas cercanas a 'fecha_pago' (target_days).
            """
            # data: (index, importe_cent, days_from_base)
            data = list(zip(df_filtrado.index.tolist(),
                            df_filtrado['IMPORTE_CENT'].astype(int).tolist(),
                            df_filtrado['DAYS_FROM_BASE'].astype(int).tolist()))
            n = len(data)
            model = cp_model.CpModel()
            x = [model.NewBoolVar(f"sel_{i}") for i in range(n)]

            # (1) Suma exacta
            model.Add(sum(x[i] * data[i][1] for i in range(n)) == int(objetivo_cent))

            # Coste de cercanía a fecha objetivo (precomputado, lineal)
            if target_days is not None:
                costs = [abs(data[i][2] - target_days) for i in range(n)]
                max_cost = (max(costs) if costs else 0) * n
                BIG_M = max_cost + 1  # Garantiza prioridad a minimizar número de facturas

                # (2)+(3) Objetivo lexicográfico: primero nº facturas, luego cercanía fechas
                model.Minimize(BIG_M * sum(x) + sum(x[i] * costs[i] for i in range(n)))
            else:
                # Solo minimizar número de facturas si no hay fecha de pago
                model.Minimize(sum(x))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10
            # Puedes paralelizar si quieres: solver.parameters.num_search_workers = 8
            status = solver.Solve(model)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                seleccionadas_idx = [data[i][0] for i in range(n) if solver.Value(x[i]) == 1]
                return seleccionadas_idx
            else:
                return None

        # --- Llamada ---
        target_days = None
        if fecha_pago:
            try:
                # fecha_pago es date; convierto a datetime para restar
                target_days = (pd.to_datetime(datetime.combine(fecha_pago, datetime.min.time())) - fecha_base).days
            except Exception:
                target_days = None

        seleccion_idx = seleccionar_facturas_exactas_ortools(df_positivas, importe_objetivo_cent, target_days)

        if seleccion_idx:
            st.success(f"✅ Combinación encontrada para {importe_objetivo_eur:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

            # Mostrar todas las columnas originales
            df_sel = df_positivas.loc[seleccion_idx].copy()

            # Orden opcional por fecha y factura si existen
            try:
                df_sel = df_sel.sort_values([col_fecha_emision, col_factura])
            except Exception:
                pass

            # Checksum visual
            suma_sel = float(df_sel['IMPORTE_CORRECTO'].sum())
            st.write(f"**Suma seleccionada:** {suma_sel:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

            st.dataframe(df_sel)

            # Descargar resultados con todas las columnas
            buffer = BytesIO()
            df_sel.to_excel(buffer, index=False, engine="openpyxl")
            buffer.seek(0)
            st.download_button(
                label="📥 Descargar facturas seleccionadas",
                data=buffer,
                file_name="facturas_seleccionadas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("❌ No se encontró una combinación EXACTA de facturas.")
