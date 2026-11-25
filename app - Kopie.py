import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------------
# Page and model configuration
# -----------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🌳",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "random_forest_model.pkl"

FEATURE_NAMES = [
    "Age",
    "NumOfProducts",
    "EstimatedSalary",
    "Balance",
    "CreditScore",
    "Tenure",
]


# -----------------------------------
# Load the model only once
# -----------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except FileNotFoundError:
    st.error("The random_forest_model.pkl file was not found.")
    st.info("Place the model file in the same directory as app.py.")
    st.stop()
except Exception as error:
    st.error(f"Error loading the model: {error}")
    st.stop()


def get_exit_probabilities(input_data):
    """Return the probability of class 1 when the model supports it."""
    if not hasattr(model, "predict_proba"):
        return None

    model_classes = list(model.classes_)
    if 1 not in model_classes:
        return None

    exited_index = model_classes.index(1)
    return model.predict_proba(input_data)[:, exited_index]


# -----------------------------------
# Application title
# -----------------------------------
st.title("🌳 Customer Churn Prediction")
st.write("Enter information for one or more customers to predict their likelihood of churn.")

single_tab, batch_tab = st.tabs([
    "👤 Single Customer Prediction",
    "📄 Batch Prediction with CSV",
])


# ===================================
# Single customer prediction
# ===================================
with single_tab:
    with st.form("prediction_form"):
        age = st.number_input(
            "Age", min_value=0, max_value=120, value=40, step=1
        )
        number_of_products = st.number_input(
            "Number of Products", min_value=1, max_value=10, value=1, step=1
        )
        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=50000.0, step=1000.0
        )
        balance = st.number_input(
            "Account Balance", min_value=0.0, value=50000.0, step=1000.0
        )
        credit_score = st.number_input(
            "Credit Score", min_value=0, max_value=1000, value=600, step=1
        )
        tenure = st.number_input(
            "Tenure", min_value=0, max_value=100, value=5, step=1
        )

        submit_button = st.form_submit_button("Predict", type="primary")

    if submit_button:
        input_data = pd.DataFrame(
            [[
                age,
                number_of_products,
                estimated_salary,
                balance,
                credit_score,
                tenure,
            ]],
            columns=FEATURE_NAMES,
        )

        try:
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("⚠️ Result: The customer is likely to churn.")
            else:
                st.success("✅ Result: The customer is likely to stay.")

            probabilities = get_exit_probabilities(input_data)
            if probabilities is not None:
                exit_probability = float(probabilities[0])
                st.metric(
                    label="Churn Probability",
                    value=f"{exit_probability:.2%}",
                )
                st.progress(exit_probability)

            with st.expander("View Entered Information"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as error:
            st.error(f"Error during prediction: {error}")


# ===================================
# Batch prediction using a CSV file
# ===================================
with batch_tab:
    st.subheader("Upload Customer Data")
    st.write(
        "Each row must represent one customer, and the file must contain the following columns:"
    )
    st.code(", ".join(FEATURE_NAMES))

    # Sample data demonstrating the required CSV structure
    sample_data = pd.DataFrame({
        "Age": [40, 52, 31],
        "NumOfProducts": [1, 2, 1],
        "EstimatedSalary": [50000.0, 82000.0, 42000.0],
        "Balance": [60000.0, 125000.0, 0.0],
        "CreditScore": [650, 580, 720],
        "Tenure": [5, 8, 2],
    })

    st.download_button(
        label="⬇️ Download Sample CSV Template",
        data=sample_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="customer_input_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Select a CSV File",
        type=["csv"],
        help="The file can contain 10, 100, or more customers.",
    )

    if uploaded_file is not None:
        try:
            customer_data = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Unable to read the CSV file: {error}")
        else:
            st.write(f"Number of customers in the file: **{len(customer_data)}**")
            st.dataframe(customer_data.head(20), use_container_width=True)

            missing_columns = [
                column for column in FEATURE_NAMES
                if column not in customer_data.columns
            ]

            if customer_data.empty:
                st.error("The CSV file is empty.")
            elif missing_columns:
                st.error(
                    "The following columns are missing from the file: "
                    + ", ".join(missing_columns)
                )
            elif st.button("🔍 Predict for All Customers", type="primary"):
                # Use only the model features in the same order as during training.
                model_input = customer_data[FEATURE_NAMES].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                invalid_rows = model_input.isna().any(axis=1)

                if invalid_rows.any():
                    # CSV data row numbers start at 2 because the first row contains the header.
                    invalid_csv_rows = (model_input.index[invalid_rows] + 2).tolist()
                    shown_rows = invalid_csv_rows[:20]
                    row_text = ", ".join(map(str, shown_rows))

                    st.error(
                        "Missing or nonnumeric values were found in the required columns. "
                        f"CSV row numbers containing errors: {row_text}"
                    )
                    if len(invalid_csv_rows) > 20:
                        st.warning(
                            f"{len(invalid_csv_rows) - 20} additional rows contain errors."
                        )
                else:
                    try:
                        predictions = model.predict(model_input)
                        result_data = customer_data.copy()
                        result_data["Prediction"] = predictions
                        result_data["Churn_Status"] = [
                            "Exited" if prediction == 1 else "Stayed"
                            for prediction in predictions
                        ]

                        probabilities = get_exit_probabilities(model_input)
                        if probabilities is not None:
                            result_data["Exit_Probability"] = probabilities.round(4)

                        exited_count = int((predictions == 1).sum())
                        stayed_count = int((predictions == 0).sum())

                        st.success(
                            f"Predictions successfully completed for {len(result_data)} customers."
                        )

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Customers", len(result_data))
                        col2.metric("Exited", exited_count)
                        col3.metric("Stayed", stayed_count)

                        st.dataframe(result_data, use_container_width=True)

                        result_csv = result_data.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="⬇️ Download Prediction Results",
                            data=result_csv,
                            file_name="customer_churn_predictions.csv",
                            mime="text/csv",
                        )

                    except Exception as error:
                        st.error(f"Error during batch prediction: {error}")
