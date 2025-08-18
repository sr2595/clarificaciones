import streamlit as st
import pandas as pd
from io import BytesIO
from ortools.sat.python import cp_model
from datetime import datetime

# --- Configuración inicial ---
st.set_page_config(page_title="UTE Clarificador Final", layout="wide")
st.title("📊 UTE Clarificador Final")

# --- Subida de archivo ---
archivo = st.file_uploader("📂 Sube un archivo Excel con las facturas", type=["xlsx"])

# --- Inicializar variables para evitar NameError ---
factura_final = None
df_internas = pd.DataFrame()
df_resultado = pd.DataFrame()

if archivo:
    # --- Cargar Excel ---
    df = pd.read_excel(archivo)

    # --- Normalizar nombres de columnas ---
    df.columns = [c.strip().upper() for c in df.columns]

    # --- Definir columnas importantes ---
    col_factura = "FACTURA_ID"
    col_cif = "CIF"
    col_nombre_cliente = "NOMBRE_CLIENTE"
    col_grupo = "GRUPO_CLIENTE"
    col_fecha_emision = "FECHA_EMISION"
    col_sociedad = "SOCIEDAD"

    # --- Preprocesamiento ---
    df["IMPORTE_CORRECTO"] = pd.to_numeric(df["IMPORTE_CORRECTO"], errors="coerce").fillna(0)
    df = df[df["IMPORTE_CORRECTO"] > 0].copy()  # 🔴 Eliminar importes negativos
    df["IMPORTE_CENT"] = (df["IMPORTE_CORRECTO"] * 100).astype(int)

    if col_fecha_emision in df.columns:
        df[col_fecha_emision] = pd.to_datetime(df[col_fecha_emision], errors="coerce")

    # --- Seleccionar cliente final ---
    clientes_finales = df[df["ES_CLIENTE_FINAL"] == True][[col_grupo]].drop_duplicates()
    cliente_final_display = st.selectbox("Selecciona el cliente final", clientes_finales[col_grupo])
    cliente_final_grupo = cliente_final_display

    # --- Selección factura final ---
    df_tss = df[(df[col_grupo] == cliente_final_grupo) & (df[col_sociedad] == "TSS")]
    if not df_tss.empty:
        facturas_finales = [
            f"{row[col_factura]} - {row['IMPORTE_CORRECTO']:,.2f} €"
            for _, row in df_tss.iterrows()
        ]
        factura_final_display = st.selectbox("Selecciona factura TSS (final)", facturas_finales)
        factura_final_id = factura_final_display.split(" - ")[0]
        factura_final = df_tss[df_tss[col_factura] == factura_final_id].iloc[0]

        st.info(
            f"Factura final seleccionada: **{factura_final[col_factura]}** "
            f"({factura_final['IMPORTE_CORRECTO']:,.2f} €)"
        )
    else:
        st.warning("⚠️ No hay facturas TSS disponibles para seleccionar")
        factura_final = None

    # --- Filtrar UTES del mismo grupo ---
    df_utes_grupo = df[(df[col_grupo] == cliente_final_grupo) & (df["ES_UTE"])].copy()

    if df_utes_grupo.empty:
        st.warning("⚠️ No hay UTES disponibles para este cliente final")
    else:
        # --- Crear lista de socios únicos ---
        df_utes_grupo_sorted = (
            df_utes_grupo[[col_cif, col_nombre_cliente]]
            .drop_duplicates()
            .sort_values(by=col_cif)
        )
        opciones_utes = [
            f"{row[col_cif]} - {row[col_nombre_cliente]}"
            if row[col_nombre_cliente]
            else f"{row[col_cif]}"
            for _, row in df_utes_grupo_sorted.iterrows()
        ]
        mapping_utes_cif = dict(zip(opciones_utes, df_utes_grupo_sorted[col_cif]))

        socios_display = st.multiselect("Selecciona CIF(s) de la UTE (socios)", opciones_utes)
        socios_cifs = [mapping_utes_cif[s] for s in socios_display]
        df_internas = df[df[col_cif].isin(socios_cifs)].copy()

        # --- Solver ---
        def cuadrar_internas(externa, df_internas, tol=100):
            """tol en céntimos, default 1€ = 100 céntimos"""
            if externa is None or df_internas.empty:
                return pd.DataFrame()

            objetivo = int(externa["IMPORTE_CENT"])
            fecha_ref = externa[col_fecha_emision]

            data = list(
                zip(
                    df_internas.index.tolist(),
                    df_internas["IMPORTE_CENT"].astype(int).tolist(),
                    (df_internas[col_fecha_emision] - fecha_ref)
                    .dt.days.fillna(0)
                    .astype(int)
                    .tolist(),
                    df_internas[col_sociedad].tolist(),
                )
            )

            n = len(data)
            if n == 0:
                return pd.DataFrame()

            model = cp_model.CpModel()
            x = [model.NewBoolVar(f"sel_{i}") for i in range(n)]

            # --- Restricción de importe ---
            model.Add(sum(x[i] * data[i][1] for i in range(n)) >= objetivo - tol)
            model.Add(sum(x[i] * data[i][1] for i in range(n)) <= objetivo + tol)

            # --- Restricción: solo una factura por sociedad ---
            sociedades = {}
            for i in range(n):
                sociedad = data[i][3]
                if sociedad not in sociedades:
                    sociedades[sociedad] = []
                sociedades[sociedad].append(x[i])
            for sociedad, vars_soc in sociedades.items():
                model.Add(sum(vars_soc) <= 1)

            # --- Objetivo ---
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

        # --- Ejecutar solver ---
        if factura_final is not None and not df_internas.empty:
            df_resultado = cuadrar_internas(factura_final, df_internas)
            if df_resultado.empty:
                st.warning("❌ No se encontró combinación de facturas internas que cuadre con la factura externa")
            else:
                st.success(
                    f"✅ Se han seleccionado {len(df_resultado)} factura(s) interna(s) que cuadran con la externa"
                )

                # --- Mostrar tabla final ---
                st.dataframe(
                    df_resultado[
                        [
                            col_factura,
                            col_cif,
                            col_nombre_cliente,
                            "IMPORTE_CORRECTO",
                            col_fecha_emision,
                            col_sociedad,
                        ]
                    ]
                )

                # --- Botón de descarga ---
                def to_excel(df_out):
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df_out.to_excel(writer, index=False, sheet_name="Resultado")
                    processed_data = output.getvalue()
                    return processed_data

                excel_data = to_excel(df_resultado)
                st.download_button(
                    label="📥 Descargar Excel con facturas internas seleccionadas",
                    data=excel_data,
                    file_name=f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
