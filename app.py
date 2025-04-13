import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import io
import base64

from src.models.first_order import first_order_model, fit_first_order
from src.models.second_order import second_order_model, fit_second_order
from src.models.langmuir_hinshelwood import langmuir_hinshelwood_model, fit_langmuir_hinshelwood
from src.utils.data_processing import parse_csv_data, parse_text_data, validate_data, calculate_statistics
from src.utils.export import export_to_csv, export_plot_to_png, generate_report

# Page configuration
st.set_page_config(
    page_title="Pollutant Degradation Kinetics Visualizer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Pollutant Degradation Kinetics Visualizer")
st.markdown("""
This application helps visualize and analyze pollutant degradation kinetics data.
Select a kinetic model and enter your experimental data below.
""")

# Sidebar for input parameters
with st.sidebar:
    st.header("Experimental Parameters")
    
    # Model selection
    model_type = st.selectbox(
        "Select Kinetic Model",
        ["First Order", "Second Order", "Langmuir-Hinshelwood"]
    )
    
    # Basic parameters
    c0 = st.number_input("Initial Concentration (C₀)", min_value=0.0, value=10.0, step=0.1)
    catalyst_loading = st.number_input("Catalyst Loading (g/L)", min_value=0.0, value=1.0, step=0.1)
    light_intensity = st.number_input("Light Intensity (mW/cm²)", min_value=0.0, value=100.0, step=1.0)
    
    # Advanced parameters
    with st.expander("Advanced Parameters"):
        h2o2_enabled = st.checkbox("Enable H₂O₂ Concentration", value=False)
        if h2o2_enabled:
            h2o2_conc = st.number_input("H₂O₂ Concentration (mM)", min_value=0.0, value=10.0, step=0.1)
        
        catalyst_type = st.selectbox(
            "Catalyst Type",
            ["TiO₂", "ZnO", "WO₃", "Custom"]
        )
        if catalyst_type == "Custom":
            catalyst_name = st.text_input("Custom Catalyst Name")

# Main content area
st.header("Data Input")
data_input_method = st.radio(
    "Select Data Input Method",
    ["Manual Entry", "CSV Upload", "Copy-Paste"]
)

time_points = None
conc_points = None

if data_input_method == "Manual Entry":
    col1, col2 = st.columns(2)
    with col1:
        time_data = st.text_area("Time (min)", "0\n5\n10\n15\n20\n30\n45\n60")
    with col2:
        conc_data = st.text_area("Concentration (mg/L)", "10\n8.5\n7.2\n6.1\n5.2\n4.1\n3.3\n2.7")
    time_points, conc_points = parse_text_data(time_data, conc_data)

elif data_input_method == "CSV Upload":
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            time_points, conc_points = parse_csv_data(uploaded_file.getvalue().decode())
        except Exception as e:
            st.error(f"Error parsing CSV file: {str(e)}")

elif data_input_method == "Copy-Paste":
    data_text = st.text_area("Paste data (time,concentration pairs, one per line)", 
                            "0,10\n5,8.5\n10,7.2\n15,6.1\n20,5.2\n30,4.1\n45,3.3\n60,2.7")
    try:
        data = [line.split(',') for line in data_text.split('\n') if line.strip()]
        time_points = np.array([float(row[0]) for row in data])
        conc_points = np.array([float(row[1]) for row in data])
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")

# Process and display results
if time_points is not None and conc_points is not None:
    # Validate data
    errors = validate_data(time_points, conc_points)
    if errors:
        for error in errors:
            st.error(error)
    else:
        # Fit selected model
        if model_type == "First Order":
            k, r2, rmse = fit_first_order(time_points, conc_points, c0)
            predicted = first_order_model(time_points, k, c0)
            parameters = {'k': k}
            
        elif model_type == "Second Order":
            k, r2, rmse = fit_second_order(time_points, conc_points, c0)
            predicted = second_order_model(time_points, k, c0)
            parameters = {'k': k}
            
        else:  # Langmuir-Hinshelwood
            k, K, r2, rmse = fit_langmuir_hinshelwood(time_points, conc_points, c0)
            predicted = langmuir_hinshelwood_model(time_points, k, K, c0)
            parameters = {'k': k, 'K': K}
        
        # Calculate additional statistics
        stats = calculate_statistics(predicted, conc_points)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Main plot
        ax1.scatter(time_points, conc_points, label='Experimental Data', color='blue')
        t_fit = np.linspace(0, max(time_points), 100)
        if model_type == "First Order":
            c_fit = first_order_model(t_fit, k, c0)
        elif model_type == "Second Order":
            c_fit = second_order_model(t_fit, k, c0)
        else:
            c_fit = langmuir_hinshelwood_model(t_fit, k, K, c0)
        ax1.plot(t_fit, c_fit, 'r-', label='Fitted Curve')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Concentration (mg/L)')
        ax1.set_title(f'{model_type} Kinetic Model Fit')
        ax1.legend()
        ax1.grid(True)
        
        # Residual plot
        residuals = conc_points - predicted
        ax2.scatter(time_points, residuals, color='green')
        ax2.axhline(y=0, color='r', linestyle='-')
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Residuals (mg/L)')
        ax2.set_title('Residual Plot')
        ax2.grid(True)
        
        # Display plots
        st.pyplot(fig)
        
        # Display results
        st.header("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Parameters")
            for param, value in parameters.items():
                st.metric(param, f"{value:.4f}")
        with col2:
            st.subheader("Fit Statistics")
            st.metric("R²", f"{r2:.4f}")
            st.metric("RMSE", f"{rmse:.4f} mg/L")
            st.metric("MAE", f"{stats['mae']:.4f} mg/L")
        
        # Export options
        st.header("Export Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Data to CSV"):
                csv = export_to_csv(time_points, conc_points, predicted)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="kinetics_data.csv",
                    mime="text/csv"
                )
        with col2:
            if st.button("Export Plot to PNG"):
                png = export_plot_to_png(fig)
                st.download_button(
                    label="Download Plot",
                    data=png,
                    file_name="kinetics_plot.png",
                    mime="image/png"
                )
        
        # Generate and display report
        results = {
            'parameters': parameters,
            'statistics': stats
        }
        report = generate_report(results)
        st.text_area("Analysis Report", report, height=200) 