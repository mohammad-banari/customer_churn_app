import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------------
# تنظیمات صفحه
# -----------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🌳",
    layout="centered"
)


# -----------------------------------
# مسیر فایل مدل
# -----------------------------------
MODEL_PATH = Path(__file__).parent / "random_forest_model.pkl"


# -----------------------------------
# بارگذاری مدل
# -----------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model


try:
    model = load_model()

except FileNotFoundError:
    st.error("فایل random_forest_model.pkl پیدا نشد.")
    st.info("فایل مدل را در همان پوشه‌ای قرار دهید که app.py قرار دارد.")
    st.stop()

except Exception as error:
    st.error(f"خطا در بارگذاری مدل: {error}")
    st.stop()


# -----------------------------------
# عنوان اپلیکیشن
# -----------------------------------
st.title("🌳 Customer Churn Prediction")

st.write(
    "اطلاعات مشتری را وارد کنید تا مدل احتمال خروج مشتری را پیش‌بینی کند."
)


# -----------------------------------
# دریافت اطلاعات کاربر
# -----------------------------------
with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=40,
        step=1
    )

    number_of_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        max_value=1000,
        value=600,
        step=1
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    submit_button = st.form_submit_button(
        "Predict",
        type="primary"
    )


# -----------------------------------
# انجام پیش‌بینی
# -----------------------------------
if submit_button:

    # نام و ترتیب ستون‌ها باید دقیقاً مشابه زمان آموزش باشند
    feature_names = [
        "Age",
        "NumOfProducts",
        "EstimatedSalary",
        "Balance",
        "CreditScore",
        "Tenure"
    ]

    input_data = pd.DataFrame(
        [[
            age,
            number_of_products,
            estimated_salary,
            balance,
            credit_score,
            tenure
        ]],
        columns=feature_names
    )

    try:
        prediction = model.predict(input_data)[0]

        # نمایش نتیجه
        if prediction == 1:
            st.error("⚠️ نتیجه: احتمال خروج مشتری وجود دارد.")
        else:
            st.success("✅ نتیجه: احتمالاً مشتری باقی می‌ماند.")

        # نمایش احتمال خروج
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]
            model_classes = list(model.classes_)

            if 1 in model_classes:
                exited_index = model_classes.index(1)
                exit_probability = probabilities[exited_index]

                st.metric(
                    label="احتمال خروج مشتری",
                    value=f"{exit_probability:.2%}"
                )

                st.progress(float(exit_probability))

        # نمایش اطلاعات واردشده
        with st.expander("نمایش اطلاعات واردشده"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"خطا هنگام پیش‌بینی: {error}")