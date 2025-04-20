import streamlit as st
import pandas as pd
# import zipfile # No longer needed
# import os      # No longer needed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Set up page
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📉 Telco Customer Churn Predictor")

# Load data directly from CSV
def load_data():
    # Define the path directly to the CSV file in the current folder
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    try:
        df = pd.read_csv(csv_path)
        # Data cleaning steps from original function
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{csv_path}' was not found in the current directory.")
        st.stop() # Stop the script if the file isn't found
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop() # Stop on other potential errors during loading

@st.cache_data
def train_model(_df): # Changed input name slightly to avoid conflict if needed, common practice with decorators
    df_model = _df.drop(columns=['customerID'])
    categorical_cols = df_model.select_dtypes(include='object').columns.drop('Churn') # Exclude Churn if present
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle potential unseen values during prediction later if needed
        # For training, fit_transform is fine
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    # Encode Churn separately
    if 'Churn' in df_model.columns:
        le_churn = LabelEncoder()
        df_model['Churn'] = le_churn.fit_transform(df_model['Churn'])
        label_encoders['Churn'] = le_churn # Store churn encoder if needed elsewhere, though often just used for y

    X = df_model.drop(columns=['Churn'])
    y = df_model['Churn']

    # Ensure columns for scaling exist
    num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if not all(col in X.columns for col in num_cols_to_scale):
        st.error(f"Error: One or more numeric columns {num_cols_to_scale} not found for scaling.")
        st.stop()
        
    scaler = StandardScaler()
    X[num_cols_to_scale] = scaler.fit_transform(X[num_cols_to_scale])

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    # Return feature names in the order the model was trained on
    return model, label_encoders, scaler, X.columns.tolist() 

# Load data and model
df = load_data()
# Ensure df is not None before proceeding (load_data handles errors with st.stop)
if df is not None:
    model, label_encoders, scaler, feature_names = train_model(df.copy()) # Pass a copy to avoid modifying original df

    st.sidebar.header("🧾 Customer Input")

    def get_user_input():
        # Use unique keys for widgets if they might appear elsewhere or be dynamically generated
        gender = st.sidebar.selectbox("Gender", options=label_encoders['gender'].classes_, key='gender_input')
        # SeniorCitizen is already numeric 0/1 in original data loading (after coerce/dropna), but let's treat input consistently
        senior_map = {"Yes": 1, "No": 0}
        senior_selection = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"], key='senior_input')
        senior = senior_map[senior_selection]

        partner = st.sidebar.selectbox("Has Partner?", options=label_encoders['Partner'].classes_, key='partner_input')
        dependents = st.sidebar.selectbox("Has Dependents?", options=label_encoders['Dependents'].classes_, key='dependents_input')
        tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12, key='tenure_input')
        phone = st.sidebar.selectbox("Phone Service", options=label_encoders['PhoneService'].classes_, key='phone_input')
        multiple = st.sidebar.selectbox("Multiple Lines", options=label_encoders['MultipleLines'].classes_, key='multiple_input')
        internet = st.sidebar.selectbox("Internet Service", options=label_encoders['InternetService'].classes_, key='internet_input')
        online_sec = st.sidebar.selectbox("Online Security", options=label_encoders['OnlineSecurity'].classes_, key='onlinesec_input')
        online_backup = st.sidebar.selectbox("Online Backup", options=label_encoders['OnlineBackup'].classes_, key='onlinebackup_input')
        protection = st.sidebar.selectbox("Device Protection", options=label_encoders['DeviceProtection'].classes_, key='protection_input')
        tech = st.sidebar.selectbox("Tech Support", options=label_encoders['TechSupport'].classes_, key='tech_input')
        tv = st.sidebar.selectbox("Streaming TV", options=label_encoders['StreamingTV'].classes_, key='tv_input')
        movies = st.sidebar.selectbox("Streaming Movies", options=label_encoders['StreamingMovies'].classes_, key='movies_input')
        contract = st.sidebar.selectbox("Contract", options=label_encoders['Contract'].classes_, key='contract_input')
        paperless = st.sidebar.selectbox("Paperless Billing", options=label_encoders['PaperlessBilling'].classes_, key='paperless_input')
        payment = st.sidebar.selectbox("Payment Method", options=label_encoders['PaymentMethod'].classes_, key='payment_input')
        charges = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 65.0, key='charges_input')
        # Calculate TotalCharges based on tenure and monthly charges for consistency
        # Note: This might differ slightly from the original dataset's TotalCharges for existing customers
        total = float(charges * tenure) 

        # Create dictionary matching the order expected by the model (feature_names)
        # Important: The order MUST match feature_names from train_model
        input_dict = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_backup,
            'DeviceProtection': protection,
            'TechSupport': tech,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': charges,
            'TotalCharges': total
        }
        
        # Ensure the dict has columns in the same order as feature_names
        # Note: This assumes feature_names includes all keys from input_dict
        # and doesn't include 'Churn' or 'customerID'
        ordered_input = {col: input_dict[col] for col in feature_names}

        return pd.Series(ordered_input)


    user_input = get_user_input()

    # Encode user input
    def prepare_input(user_input_series, label_encoders_map, scaler_obj, feature_order):
        # Convert Series to DataFrame to preserve column names/order easily
        input_df = pd.DataFrame([user_input_series])
        
        # Apply label encoding
        for col, le in label_encoders_map.items():
            # Apply encoding only to columns present in the input AND that are categorical (object type originally)
            # Check if column requires encoding based on original dtype or presence in label_encoders
            if col in input_df.columns and col not in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']: # Exclude numericals + target
                 # Use try-except for robustness against potentially unseen values in selectboxes
                 try:
                     input_df[col] = le.transform(input_df[col])
                 except ValueError:
                     st.warning(f"Value '{input_df[col].iloc[0]}' in '{col}' was not seen during training. Prediction may be unreliable.")
                     # Handle unseen value: e.g., assign -1 or use a default category encoding
                     # For simplicity, we might let the model handle it or default, but flagging is good.
                     # Or find the encoding for a default like 'No' or 'No internet service' if applicable
                     default_val = 'No' # Example default
                     if default_val in le.classes_:
                          input_df[col] = le.transform([default_val])[0]
                     else: 
                          input_df[col] = -1 # Or another placeholder like np.nan before scaling

        # Apply scaling to numerical columns
        num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        # Ensure TotalCharges is float before scaling
        input_df['TotalCharges'] = input_df['TotalCharges'].astype(float) 
        input_df[num_cols_to_scale] = scaler_obj.transform(input_df[num_cols_to_scale])

        # Ensure final DataFrame columns are in the exact order the model expects
        input_df = input_df[feature_order]

        return input_df.values # Return as numpy array for prediction


    # Predict
    # Ensure all necessary components are available from training step
    if 'model' in locals() and 'label_encoders' in locals() and 'scaler' in locals() and 'feature_names' in locals():
        prepared_input = prepare_input(user_input, label_encoders, scaler, feature_names)
        proba = model.predict_proba(prepared_input)[0][1] # Probability of class '1' (Churn)

        # Display result
        st.markdown("### 🔮 Prediction")
        st.metric(label="Churn Probability", value=f"{proba:.2%}", delta=None)

        if proba > 0.5:
            st.error("⚠️ High Risk of Churn")
        else:
            st.success("✅ Low Risk of Churn")

    else:
        st.error("Model components not loaded correctly. Please check the training process.")


    st.markdown("---")
    # Updated caption
    st.caption("Place the 'WA_Fn-UseC_-Telco-Customer-Churn.csv' file and run this script in the same folder. Uses a Random Forest model.")

else:
    st.error("Failed to load data. Cannot proceed.")