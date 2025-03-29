import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Predictor import DiabetesPredictor

# Set page config
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the predictor
@st.cache_resource
def load_predictor():
    predictor = DiabetesPredictor(models_dir='models')
    predictor.load_models()
    return predictor

predictor = load_predictor()
available_models = predictor.get_available_models()

# Add 'All Models' option
model_options = ['All Models'] + available_models

# Title and description
st.title("Diabetes Risk Prediction System")
st.markdown("This application uses machine learning to predict diabetes risk based on health indicators from the CDC's BRFSS dataset.")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Comparison"])

# Home page
if page == "Home":
    st.header("Welcome to the Diabetes Prediction Tool")
    
    st.subheader("About This Project")
    st.write("""
    This tool uses machine learning models trained on the CDC's Behavioral Risk Factor Surveillance System (BRFSS) dataset 
    to predict an individual's risk of diabetes or prediabetes based on health indicators.
    
    Our models have achieved accuracy rates of approximately 75%, with AUC scores of around 0.83, 
    indicating good predictive performance.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available Models")
        for model in available_models:
            st.markdown(f"✓ **{model.upper()}**")
        
        st.markdown("The system also provides an **ensemble** prediction by combining all models.")
    
    with col2:
        st.subheader("Key Risk Factors")
        st.markdown("""
        Based on our models, the top risk factors for diabetes include:
        - **General Health Status**
        - **High Blood Pressure**
        - **Body Mass Index (BMI)**
        - **Age**
        - **High Cholesterol**
        """)
    
    st.info("To use this tool, navigate to the 'Prediction' page using the sidebar and fill in your health information.")

# Prediction page
elif page == "Prediction":
    st.header("Diabetes Risk Assessment")
    st.write("Please fill in the form below with your health information to get a prediction.")
    
    # Select model to use
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    model_to_use = None if selected_model == 'All Models' else selected_model
    
    # Form for input data
    with st.form("prediction_form"):
        # Create tabs for different categories of inputs
        tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Health Measurements", "Medical Conditions", "Lifestyle"])
        
        with tab1:
            st.subheader("Demographics")
            col1, col2 = st.columns(2)
            with col1:
                age = st.selectbox("Age Category", 
                               options=range(1, 14),
                               format_func=lambda x: {
                                   1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39",
                                   5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59",
                                   9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79",
                                   13: "80+"
                               }[x],
                               index=3)
                education = st.selectbox("Education Level",
                                    options=range(1, 7),
                                    format_func=lambda x: {
                                        1: "Never attended school or only kindergarten",
                                        2: "Grades 1-8 (Elementary)",
                                        3: "Grades 9-11 (Some high school)",
                                        4: "Grade 12 or GED (High school graduate)",
                                        5: "College 1-3 years (Some college)",
                                        6: "College 4 years or more (College graduate)"
                                    }[x],
                                    index=4)
            with col2:
                sex = st.radio("Sex", ["Female", "Male"], horizontal=True, index=0)
                income = st.selectbox("Income Level",
                                 options=range(1, 9),
                                 format_func=lambda x: {
                                     1: "Less than $10,000",
                                     2: "$10,000 to $15,000",
                                     3: "$15,000 to $20,000",
                                     4: "$20,000 to $25,000",
                                     5: "$25,000 to $35,000",
                                     6: "$35,000 to $50,000",
                                     7: "$50,000 to $75,000",
                                     8: "$75,000 or more"
                                 }[x],
                                 index=5)
        
        with tab2:
            st.subheader("Health Measurements")
            col1, col2 = st.columns(2)
            with col1:
                bmi = st.number_input("BMI (Body Mass Index)", 
                                  min_value=10.0, max_value=99.0, 
                                  value=25.0, step=0.1,
                                  help="If you don't know your BMI, you can calculate it by dividing your weight (kg) by your height squared (m²)")
                phys_hlth = st.slider("Physical Health Not Good (Days in past 30 days)", 
                                  min_value=0, max_value=30, value=0,
                                  help="Number of days physical health was not good in the past 30 days")
            with col2:
                gen_hlth = st.selectbox("General Health",
                                    options=range(1, 6),
                                    format_func=lambda x: {
                                        1: "Excellent", 2: "Very good", 3: "Good", 
                                        4: "Fair", 5: "Poor"
                                    }[x],
                                    index=1)
                ment_hlth = st.slider("Mental Health Not Good (Days in past 30 days)", 
                                  min_value=0, max_value=30, value=0,
                                  help="Number of days mental health was not good in the past 30 days")
        
        with tab3:
            st.subheader("Medical Conditions")
            col1, col2 = st.columns(2)
            with col1:
                high_bp = st.radio("High Blood Pressure", ["No", "Yes"], horizontal=True, index=0)
                stroke = st.radio("Ever had a Stroke", ["No", "Yes"], horizontal=True, index=0)
                diff_walk = st.radio("Difficulty Walking", ["No", "Yes"], horizontal=True, index=0)
            with col2:
                high_chol = st.radio("High Cholesterol", ["No", "Yes"], horizontal=True, index=0)
                heart_disease = st.radio("Heart Disease or Heart Attack", ["No", "Yes"], horizontal=True, index=0)
                chol_check = st.radio("Cholesterol Check in past 5 years", ["No", "Yes"], horizontal=True, index=1)
        
        with tab4:
            st.subheader("Lifestyle Factors")
            col1, col2 = st.columns(2)
            with col1:
                smoker = st.radio("Smoker", ["No", "Yes"], horizontal=True, index=0)
                fruits = st.radio("Consume Fruits 1+ times per day", ["No", "Yes"], horizontal=True, index=1)
                alcohol = st.radio("Heavy Alcohol Consumption", ["No", "Yes"], horizontal=True, index=0)
            with col2:
                phys_activity = st.radio("Physical Activity in past 30 days", ["No", "Yes"], horizontal=True, index=1)
                veggies = st.radio("Consume Vegetables 1+ times per day", ["No", "Yes"], horizontal=True, index=1)
                healthcare = st.radio("Any Healthcare Coverage", ["No", "Yes"], horizontal=True, index=1)
                no_doc_cost = st.radio("Couldn't See Doctor Due to Cost", ["No", "Yes"], horizontal=True, index=0)
        
        # Submit button
        submit_button = st.form_submit_button("Predict Diabetes Risk")
    
    # Process form data when submitted
    if submit_button:
        # Prepare input data
        input_data = {
            'HighBP': 1 if high_bp == "Yes" else 0,
            'HighChol': 1 if high_chol == "Yes" else 0,
            'CholCheck': 1 if chol_check == "Yes" else 0,
            'BMI': bmi,
            'Smoker': 1 if smoker == "Yes" else 0,
            'Stroke': 1 if stroke == "Yes" else 0,
            'HeartDiseaseorAttack': 1 if heart_disease == "Yes" else 0,
            'PhysActivity': 1 if phys_activity == "Yes" else 0,
            'Fruits': 1 if fruits == "Yes" else 0,
            'Veggies': 1 if veggies == "Yes" else 0,
            'HvyAlcoholConsump': 1 if alcohol == "Yes" else 0,
            'AnyHealthcare': 1 if healthcare == "Yes" else 0,
            'NoDocbcCost': 1 if no_doc_cost == "Yes" else 0,
            'GenHlth': gen_hlth,
            'MentHlth': ment_hlth,
            'PhysHlth': phys_hlth,
            'DiffWalk': 1 if diff_walk == "Yes" else 0,
            'Sex': 1 if sex == "Male" else 0,
            'Age': age,
            'Education': education,
            'Income': income
        }
        
        # Make prediction
        predictions = predictor.predict(input_data, model_to_use)
        
        # Display results
        st.header("Prediction Results")
        
        # Determine which result to display most prominently
        if model_to_use:
            main_result = predictions[model_to_use]
            main_model_name = model_to_use
        else:
            main_result = predictions['ensemble']
            main_model_name = 'Ensemble (All Models)'
        
        # Main result
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if main_result['prediction'] == 1:
                st.error(f"### {main_model_name.upper()}: Higher Risk of Diabetes")
                st.markdown(f"Probability: **{main_result['probability']:.2%}**")
                st.markdown("The model predicts you have a **higher risk** of diabetes or prediabetes.")
            else:
                st.success(f"### {main_model_name.upper()}: Lower Risk of Diabetes")
                st.markdown(f"Probability: **{main_result['probability']:.2%}**")
                st.markdown("The model predicts you have a **lower risk** of diabetes.")
        
        with col2:
            # Create a gauge chart for the probability
            fig, ax = plt.subplots(figsize=(4, 1))
            ax.barh([0], [100], color='lightgray', height=0.3)
            ax.barh([0], [main_result['probability'] * 100], 
                   color='#FF4B4B' if main_result['prediction'] == 1 else '#00CC96', 
                   height=0.3)
            ax.set_xlim(0, 100)
            ax.text(main_result['probability'] * 100, 0, f" {main_result['probability']:.2%}", 
                   va='center', ha='left', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
        
        # Display all model results if using all models
        if not model_to_use:
            st.subheader("Individual Model Predictions")
            
            cols = st.columns(len(available_models))
            
            for i, model_name in enumerate(available_models):
                with cols[i]:
                    result = predictions[model_name]
                    
                    if result['prediction'] == 1:
                        st.error(f"**{model_name.upper()}**")
                        st.markdown(f"Probability: **{result['probability']:.2%}**")
                        st.markdown("**Higher Risk**")
                    else:
                        st.success(f"**{model_name.upper()}**")
                        st.markdown(f"Probability: **{result['probability']:.2%}**")
                        st.markdown("**Lower Risk**")
        
        # Feature importance
        if model_to_use and model_to_use != 'ensemble':
            st.subheader(f"Feature Importance for {model_to_use.upper()}")
            importance_plot = predictor.generate_feature_importance_plot(model_to_use)
            if importance_plot:
                st.pyplot(importance_plot)
            else:
                st.write("Feature importance information not available for this model.")
        
        # Recommendations section
        st.subheader("Recommendations")
        
        if main_result['prediction'] == 1:
            st.markdown("#### Consider these steps to reduce your diabetes risk:")
            
            recs = []
            
            if bmi > 25:
                recs.append("- **Maintain a healthy weight** - Even a modest weight loss of 5-7% can help reduce diabetes risk.")
            
            if phys_activity == "No":
                recs.append("- **Increase physical activity** - Aim for at least 150 minutes of moderate activity per week.")
            
            if high_bp == "Yes":
                recs.append("- **Monitor blood pressure** - Work with your healthcare provider to keep your blood pressure in check.")
            
            if fruits == "No" or veggies == "No":
                recs.append("- **Improve diet** - Increase consumption of fruits, vegetables, and whole grains while reducing processed foods.")
            
            recs.append("- **Get regular check-ups** - Schedule regular screenings with your healthcare provider.")
            
            for rec in recs:
                st.markdown(rec)
        else:
            st.markdown("#### Maintain your healthy lifestyle:")
            st.markdown("- **Continue regular physical activity** - Stay active with at least 150 minutes of moderate activity per week.")
            st.markdown("- **Maintain healthy eating habits** - Focus on balanced nutrition with plenty of fruits and vegetables.")
            st.markdown("- **Schedule regular check-ups** - Even with lower risk, regular screenings are important.")
        
        st.info("This prediction is based on machine learning models and should not replace professional medical advice. If you have concerns about diabetes risk, please consult with a healthcare provider.")

# Model Comparison page
elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    
    # Display model metrics
    st.subheader("Performance Metrics")
    
    metrics_data = {
        'Model': ['XGBoost', 'Random Forest', 'Gradient Boosting', 'Logistic Regression'],
        'Accuracy': [0.747, 0.749, 0.750, 0.746],
        'Precision': [0.730, 0.728, 0.729, 0.737],
        'Recall': [0.786, 0.796, 0.796, 0.764],
        'F1 Score': [0.757, 0.761, 0.761, 0.750],
        'ROC AUC': [0.823, 0.828, 0.829, 0.823]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.set_index('Model')
    
    # Highlight the best model for each metric
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
    
    # Visualize metrics
    st.subheader("Metrics Visualization")
    
    # Reshape data for plotting
    metrics_long = pd.melt(metrics_df.reset_index(), id_vars=['Model'], 
                          value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                          var_name='Metric', value_name='Value')
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_long)
    plt.ylim(0.7, 0.85)  # Adjust y-axis for better visualization
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=0)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Feature importance comparison
    st.subheader("Feature Importance Comparison")
    
    # Create columns for each model
    cols = st.columns(len(available_models))
    
    for i, model_name in enumerate(available_models):
        with cols[i]:
            st.markdown(f"**{model_name.upper()}**")
            importance_plot = predictor.generate_feature_importance_plot(model_name, top_n=5)
            if importance_plot:
                st.pyplot(importance_plot)
            else:
                st.write("Feature importance not available")
    
    st.markdown("---")
    st.write("Based on the metrics, **Gradient Boosting** has the best overall performance, achieving the highest accuracy (0.750) and ROC AUC (0.829).")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Diabetes Risk Prediction System © 2025")