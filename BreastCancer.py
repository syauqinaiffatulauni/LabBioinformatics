import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
import pickle

# Helper functions
def preprocess_data(df, normalize=True):
    """Preprocess dataset: background normalization, mapping, z-score normalization."""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    
    # Background normalization: Replace negative values with 0 in numeric columns
    numeric_cols[numeric_cols < 0] = 0
    
    # Replace the original numeric columns with the normalized ones
    df[numeric_cols.columns] = numeric_cols
    
    # Example mapping: (you can adjust this step if needed)
    # df = map_data(df)
    
    if normalize:
        scaler = StandardScaler()
        numeric_cols = pd.DataFrame(
            scaler.fit_transform(numeric_cols),
            columns=numeric_cols.columns
        )
        df[numeric_cols.columns] = numeric_cols
    
    return df


def perform_anova(df, target):
    """Perform ANOVA test to find significant features."""
    significant_features = {}
    for column in df.columns:
        groups = [df[target == label][column] for label in np.unique(target)]
        f_stat, p_value = f_oneway(*groups)
        if p_value < 0.05:
            significant_features[column] = p_value
    return significant_features

def train_model(df, target, model_type):
    """Train Random Forest or Gradient Boosting model."""
    X = df.drop(columns=target)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        return None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, cm, report

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Streamlit UI
def main():
    st.title("Breast Cancer Classification App")

    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Choose Mode", ["Train Model", "Use Model", "Combined"])

    if tab == "Train Model":
        st.header("Train a Classification Model")
        uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:", df.head())

            if st.checkbox("Normalize Dataset (Z-score)"):
                df = preprocess_data(df)
                st.write("Normalized Data:", df.head())

            if st.checkbox("Perform ANOVA Test"):
                target_column = st.selectbox("Select Target Column", df.columns)
                significant_features = perform_anova(df, df[target_column])
                st.write("Significant Features:", significant_features)

            model_type = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Both"])

            if st.button("Train Model"):
                target_column = st.selectbox("Select Target Column for Training", df.columns)

                if model_type == "Both":
                    rf_model, rf_cm, rf_report = train_model(df, target_column, "Random Forest")
                    gb_model, gb_cm, gb_report = train_model(df, target_column, "Gradient Boosting")
                    st.write("Random Forest Confusion Matrix:", rf_cm)
                    st.write("Gradient Boosting Confusion Matrix:", gb_cm)
                    save_model(rf_model, "random_forest_model.pkl")
                    save_model(gb_model, "gradient_boosting_model.pkl")
                else:
                    model, cm, report = train_model(df, target_column, model_type)
                    st.write(f"{model_type} Confusion Matrix:", cm)
                    save_model(model, f"{model_type.lower().replace(' ', '_')}_model.pkl")

    elif tab == "Use Model":
        st.header("Use Pretrained Model for Prediction")

        uploaded_file = st.file_uploader("Upload Dataset for Prediction", type=["csv"])
        model_file = st.file_uploader("Upload Pretrained Model", type=["pkl"])

        if uploaded_file and model_file:
            df = pd.read_csv(uploaded_file)
            model = load_model(model_file)

            predictions = model.predict(df)
            st.write("Predictions:", predictions)

    elif tab == "Combined":
        st.header("Train and Predict")

        sub_tab = st.selectbox("Select Mode", ["Train", "Predict"])

        if sub_tab == "Train":
            st.subheader("Train a Classification Model")
            # Replicate Train Model tab
        elif sub_tab == "Predict":
            st.subheader("Use Pretrained Model for Prediction")
            # Replicate Use Model tab

if __name__ == "__main__":
    main()
