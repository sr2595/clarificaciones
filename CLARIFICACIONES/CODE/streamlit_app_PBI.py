import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model
from io import BytesIO

st.set_page_config(page_title="Clarificador PBI", page_icon="📄", layout="wide")

st.title("📄 Clarificador PBI")

# --- Subir archivo Excel ---
archivo = st.file_uploader("Sube el archivo Excel PBI", type=["xlsx", "xls"])
if archivo:
    df = pd.read_excel(archivo, engine="openpyxl")

    # --- Detectar columnas ---
    possible_date_cols = ['FX_EMISION', 'FECHA', 'Fecha', 'fecha']
    col_fecha_emision = next((col for col in possible_date_cols if col in df.columns), None)

    possible_factura_cols = ['FACTURA', 'Factura', 'factura']
    col_factura = next((col for col in possible_factura_cols if col in df.columns), None)

    possible_importe_cols = ['IMPORTE', 'Importe', 'importe']
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
        except:
            return None

    df['IMPORTE_CORRECTO'] = df[col_importe].apply(convertir_importe_europeo)

    total = df['IMPORTE_CORRECTO'].sum()
    minimo = df['IMPORTE_CORRECTO'].min()
    maximo = df['IMPORTE_CORRECTO'].max()

    st.write("**📊 Resumen del archivo:**")
    st.write(f"- Número total de facturas: {len(df)}")
    st.write(f"- Suma total importes: {total:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe mínimo: {minimo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    st.write(f"- Importe máximo: {maximo:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))

    # --- Inputs para búsqueda ---
    importe_objetivo = st.text_input("Introduce importe objetivo (ej: 295.206,63)")
    fecha_pago = st.date_input("Fecha de pago")

    if importe_objetivo:
        try:
            importe_objetivo_eur = float(importe_objetivo.replace('.', '').replace(',', '.'))
            importe_objetivo_cent = int(round(importe_objetivo_eur * 100))
        except:
            st.error("Formato de importe no válido.")
            st.stop()

        fecha_base = df[col_fecha_emision].min()
        df['DAYS_FROM_BASE'] = (df[col_fecha_emision] - fecha_base).dt.days.fillna(0).astype(int)

        # --- Conversión segura de importe a céntimos ---
        if df['IMPORTE_CORRECTO'].isna().any():
            st.warning(f"⚠️ Se han encontrado {df['IMPORTE_CORRECTO'].isna().sum()} filas con importe vacío. Se considerarán como 0.")

        df['IMPORTE_CENT'] = (
            df['IMPORTE_CORRECTO']
            .fillna(0)
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
            .astype(float)
            .mul(100)
            .round()
            .astype(int)
        )

        # --- Función OR-Tools ---
        def seleccionar_facturas_exactas_ortools(df, objetivo_cent):
            model = cp_model.CpModel()
            data = list(zip(df[col_factura], df['IMPORTE_CENT'], df['DAYS_FROM_BASE'], df['IMPORTE_CORRECTO']))
            vars = [model.NewBoolVar(f"sel_{i}") for i in range(len(data))]
            model.Add(sum(vars[i] * data[i][1] for i in range(len(data))) == objetivo_cent)
            model.Minimize(sum(vars))
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10
            status = solver.Solve(model)
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                seleccionadas = []
                for i, (factura, imp_cent, _, imp_eur) in enumerate(data):
                    if solver.Value(vars[i]) == 1:
                        seleccionadas.append((factura, imp_eur))
                return seleccionadas
            else:
                return None

        seleccion = seleccionar_facturas_exactas_ortools(df, importe_objetivo_cent)

        if seleccion:
            st.success(f"✅ Combinación encontrada para {importe_objetivo_eur:,.2f} €")
            df_sel = pd.DataFrame(seleccion, columns=["Factura", "Importe (€)"])
            st.dataframe(df_sel)

            buffer = BytesIO()
            df_sel.to_excel(buffer, index=False, engine="openpyxl")
            st.download_button(
                label="📥 Descargar facturas seleccionadas",
                data=buffer,
                file_name="facturas_seleccionadas_PBI.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("❌ No se encontró una combinación EXACTA de facturas.")
