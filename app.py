import streamlit as st
import numpy as np
import pickle
import pandas as pd  # 确保导入 pandas 库

# Load the model and scaler
try:
    with open(r'D:\app\logistic_Reg.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(r'D:\app\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Failed to load model files: {str(e)}")
    st.stop()

# Set the page title
st.title("Predict Pre-sarcopenia in Metabolic Syndrome")
st.markdown("Enter **Height, Waist, Thigh Length, ALP, Sex** for prediction")

# Input components with reasonable constraints
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
waist = st.number_input("Waist (cm)", min_value=50.0, max_value=200.0, value=85.0)
thigh_length = st.number_input("Thigh Length (cm)", min_value=30.0, max_value=100.0, value=50.0)
alp = st.number_input("ALP (U/L)", min_value=20.0, max_value=500.0, value=80.0)
sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')

# Prediction logic
if st.button("Predict"):
    try:
        # Input validation
        if any([val == 0 for val in [height, waist, thigh_length, alp]]):
            st.warning("Input values cannot be zero. Please check your inputs!")
            st.stop()

        # Format input data
        input_data = np.array([[height, waist, thigh_length, alp, sex]])
        
        # Standardize input
        input_scaled = scaler.transform(input_data)
        
        # Get probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)
            positive_prob = probabilities[0][1]
        else:
            st.error("This model does not support probability prediction")
            st.stop()

        # Get prediction
        prediction = model.predict(input_scaled)
        
        # Display results
        st.subheader("Prediction Result")
        threshold = 0.5
        
        if prediction[0] == 1:
            st.success(f"**Positive Prediction (Probability: {positive_prob:.2%})**")
            st.markdown("""
            **Diagnosis**: Metabolic Syndrome with Pre-sarcopenia  
            **Recommendations**: 
            - Recommend muscle mass assessment
            - Increase protein intake
            - Regular metabolic monitoring
            """)
        else:
            st.success(f"**Negative Prediction (Probability: {1 - positive_prob:.2%})**")
            st.markdown("""
            **Diagnosis**: No Metabolic Syndrome with Pre-sarcopenia  
            **Recommendations**: 
            - Maintain a healthy diet
            - Regular metabolic monitoring
            - Engage in preventive exercise
            """)
            
        # Visualization
        probabilities_dict = {
            "Positive Probability": [positive_prob],
            "Negative Probability": [1 - positive_prob]
        }

        probabilities_df = pd.DataFrame.from_dict(probabilities_dict)

        st.bar_chart(probabilities_df)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Sidebar instructions
st.sidebar.markdown("""**User Guide**:
1. Enter patient's physiological indicators
2. Click Predict to get results
3. Predictions >50% probability will show positive diagnosis

**Reference Ranges**:
- Height: Normal adult 150-200cm
- Waist: Male<95cm, Female<80cm
- ALP: Normal range 40-160 U/L
""")