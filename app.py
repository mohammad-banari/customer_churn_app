 import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq


# -----------------------------------
# Page settings
# -----------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction & Chatbot",
    page_icon="🌳",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "random_forest_model.pkl"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b"

FEATURE_NAMES = [
    "Age",
    "NumOfProducts",
    "EstimatedSalary",
    "Balance",
    "CreditScore",
    "Tenure",
]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful assistant in a customer churn prediction app. "
        "Answer clearly and in the user's language. "
        "Remember information the user shares within this conversation. "
        "You can explain machine learning and customer churn. "
        "You do not automatically have access to the prediction form, "
        "uploaded CSV files, or prediction results. "
        "Ask the user to share relevant information when needed."
    ),
}


# -----------------------------------
# Load prediction model
# -----------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# A missing prediction model does not prevent using the chatbot.
model = None
model_error = None

try:
    model = load_model()
except FileNotFoundError:
    model_error = (
        "فایل random_forest_model.pkl پیدا نشد. "
        "آن را در همان پوشه app.py قرار دهید."
    )
except Exception as error:
    model_error = f"خطا در بارگذاری مدل: {error}"


def get_exit_probabilities(input_data):
    """Return probabilities for class 1 when supported."""
    if model is None or not hasattr(model, "predict_proba"):
        return None

    model_classes = list(model.classes_)

    if 1 not in model_classes:
        return None

    exited_index = model_classes.index(1)
    return model.predict_proba(input_data)[:, exited_index]


def get_setting(name, default=None):
    """Read Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
    except FileNotFoundError:
        value = None

    return value or os.getenv(name) or default


def generate_text(chat_history):
    """Send the full conversation to Groq."""
    api_key = get_setting("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY تنظیم نشده است. "
            "کلید را در Streamlit secrets یا متغیر محیطی قرار دهید."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=get_setting("GROQ_MODEL", DEFAULT_CHAT_MODEL),
        messages=chat_history,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("پاسخ متنی از مدل دریافت نشد.")

    return answer


# -----------------------------------
# Session state
# -----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SYSTEM_MESSAGE.copy()]

if "chatbot_started" not in st.session_state:
    st.session_state.chatbot_started = False

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if "batch_source" not in st.session_state:
    st.session_state.batch_source = None


# -----------------------------------
# App title and tabs
# -----------------------------------
st.title("🌳 Customer Churn Prediction & Chatbot")

single_tab, batch_tab, chatbot_tab = st.tabs(
    [
        "👤 پیش‌بینی یک مشتری",
        "📄 پیش‌بینی گروهی با CSV",
        "💬 Chatbot",
    ]
)


# ===================================
# Single customer prediction
# ===================================
with single_tab:
    st.write("اطلاعات مشتری را وارد کنید.")

    if model is None:
        st.error(model_error)
    else:
        with st.form("prediction_form"):
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=40,
                step=1,
            )

            number_of_products = st.number_input(
                "Number of Products",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
            )

            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
            )

            balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=0,
                max_value=1000,
                value=600,
                step=1,
            )

            tenure = st.number_input(
                "Tenure",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
            )

            submitted = st.form_submit_button(
                "Predict",
                type="primary",
            )

        if submitted:
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
                    st.error("⚠️ نتیجه: احتمال خروج مشتری وجود دارد.")
                else:
                    st.success("✅ نتیجه: احتمالاً مشتری باقی می‌ماند.")

                probabilities = get_exit_probabilities(input_data)

                if probabilities is not None:
                    probability = float(probabilities[0])

                    st.metric(
                        "احتمال خروج مشتری",
                        f"{probability:.2%}",
                    )
                    st.progress(probability)

                with st.expander("نمایش اطلاعات واردشده"):
                    st.dataframe(
                        input_data,
                        use_container_width=True,
                    )

            except Exception as error:
                st.error(f"خطا هنگام پیش‌بینی: {error}")


# ===================================
# Batch prediction
# ===================================
with batch_tab:
    st.subheader("آپلود فایل مشتریان")

    if model is None:
        st.error(model_error)
    else:
        st.write(
            "هر سطر باید مربوط به یک مشتری باشد. "
            "فایل باید ستون‌های زیر را داشته باشد:"
        )
        st.code(", ".join(FEATURE_NAMES))

        sample_data = pd.DataFrame({
            "Age": [40, 52, 31],
            "NumOfProducts": [1, 2, 1],
            "EstimatedSalary": [50000.0, 82000.0, 42000.0],
            "Balance": [60000.0, 125000.0, 0.0],
            "CreditScore": [650, 580, 720],
            "Tenure": [5, 8, 2],
        })

        st.download_button(
            "⬇️ دانلود قالب نمونه CSV",
            data=sample_data.to_csv(index=False).encode("utf-8-sig"),
            file_name="customer_input_template.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader(
            "فایل CSV را انتخاب کنید",
            type=["csv"],
        )

        if uploaded_file is None:
            st.session_state.batch_source = None
            st.session_state.batch_results = None

        else:
            file_bytes = uploaded_file.getvalue()

            # Clear old results when the uploaded file changes.
            if st.session_state.batch_source != file_bytes:
                st.session_state.batch_source = file_bytes
                st.session_state.batch_results = None

            try:
                uploaded_file.seek(0)
                customer_data = pd.read_csv(uploaded_file)
            except Exception as error:
                st.error(f"فایل CSV قابل خواندن نیست: {error}")
            else:
                st.write(
                    f"تعداد مشتریان موجود در فایل: "
                    f"**{len(customer_data)}**"
                )

                st.dataframe(
                    customer_data.head(20),
                    use_container_width=True,
                )

                missing_columns = [
                    column
                    for column in FEATURE_NAMES
                    if column not in customer_data.columns
                ]

                if customer_data.empty:
                    st.error("فایل CSV خالی است.")

                elif missing_columns:
                    st.error(
                        "ستون‌های زیر وجود ندارند: "
                        + ", ".join(missing_columns)
                    )

                elif st.button(
                    "🔍 پیش‌بینی برای همه مشتریان",
                    type="primary",
                ):
                    model_input = customer_data[FEATURE_NAMES].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )

                    # Reject missing, nonnumeric, and infinite values.
                    invalid_rows = ~np.isfinite(
                        model_input.to_numpy(dtype=float)
                    ).all(axis=1)

                    if invalid_rows.any():
                        invalid_csv_rows = (
                            model_input.index[invalid_rows] + 2
                        ).tolist()

                        st.error(
                            "مقدار خالی، غیرعددی یا نامعتبر وجود دارد. "
                            "شماره سطرهای CSV: "
                            + ", ".join(
                                map(str, invalid_csv_rows[:20])
                            )
                        )

                        if len(invalid_csv_rows) > 20:
                            st.warning(
                                f"{len(invalid_csv_rows) - 20} "
                                "سطر مشکل‌دار دیگر نیز وجود دارد."
                            )

                    else:
                        try:
                            predictions = model.predict(model_input)

                            result_data = customer_data.copy()
                            result_data["Prediction"] = predictions
                            result_data["Churn_Status"] = [
                                "Exited" if value == 1 else "Stayed"
                                for value in predictions
                            ]

                            probabilities = get_exit_probabilities(
                                model_input
                            )

                            if probabilities is not None:
                                result_data["Exit_Probability"] = (
                                    probabilities.round(4)
                                )

                            st.session_state.batch_results = result_data

                        except Exception as error:
                            st.error(
                                f"خطا هنگام پیش‌بینی گروهی: {error}"
                            )

            # Keep results available across Streamlit reruns.
            results = st.session_state.batch_results

            if results is not None:
                st.success(
                    f"پیش‌بینی برای {len(results)} مشتری انجام شد."
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("تعداد کل", len(results))
                col2.metric(
                    "خروج",
                    int((results["Prediction"] == 1).sum()),
                )
                col3.metric(
                    "ماندن",
                    int((results["Prediction"] == 0).sum()),
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                )

                st.download_button(
                    "⬇️ دانلود نتایج پیش‌بینی",
                    data=results.to_csv(index=False).encode(
                        "utf-8-sig"
                    ),
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv",
                )


# ===================================
# Chatbot
# ===================================
with chatbot_tab:
    st.subheader("💬 AI Chatbot")

    st.caption(
        "گفتگو در همین نشست حفظ می‌شود. "
        "با بارگذاری مجدد صفحه یا شروع نشست جدید ممکن است پاک شود. "
        "چت‌بات به‌صورت خودکار به فایل CSV یا نتایج پیش‌بینی دسترسی ندارد."
    )

    if not st.session_state.chatbot_started:
        if st.button(
            "💬 Start Chatbot",
            type="primary",
        ):
            st.session_state.chatbot_started = True
            st.rerun()

    else:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = [
                SYSTEM_MESSAGE.copy()
            ]
            st.rerun()

        # Display previous messages, excluding the system message.
        for message in st.session_state.chat_history:
            if message["role"] in ("user", "assistant"):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input("پیام خود را بنویسید...")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)

            # Only save the exchange if the API request succeeds.
            request_history = [
                *st.session_state.chat_history,
                {"role": "user", "content": prompt},
            ]

            with st.chat_message("assistant"):
                try:
                    with st.spinner("در حال پاسخ دادن..."):
                        answer = generate_text(request_history)

                    st.markdown(answer)

                    st.session_state.chat_history = [
                        *request_history,
                        {
                            "role": "assistant",
                            "content": answer,
                        },
                    ]

                except Exception as e:
                    st.error(f"Error: {type(e).__name__}: {str(e)}")
                    st.exception(e)
                    st.error(
                        "دریافت پاسخ ناموفق بود. "
                        "کلید Groq، نام مدل، اتصال اینترنت و "
                        "محدودیت استفاده را بررسی کرده و دوباره تلاش کنید."
                    )
