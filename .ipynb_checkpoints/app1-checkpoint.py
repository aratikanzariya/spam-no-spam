
import streamlit as st
import pickle
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ---------------- MODEL PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


# ---------------- PREDICTION ----------------
def predict_message(message, model, vectorizer):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    # Support common Spam/Ham and 1/0 labels
    label = str(prediction).strip().lower()

    if label in ("spam", "1", "true"):
        result = "Spam"
    elif label in ("ham", "not spam", "not_spam", "0", "false"):
        result = "Not Spam"
    else:
        result = str(prediction)

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(message_vector)[0]
        classes = list(model.classes_)

        predicted_index = classes.index(prediction)
        confidence = float(probabilities[predicted_index]) * 100

    return result, confidence


# ---------------- APP UI ----------------
st.title("📩 SMS Spam Detector")
st.write(
    "Enter a message below to check whether it is Spam or Not Spam."
)

st.divider()

try:
    model, vectorizer = load_model()
except Exception as error:
    st.error(
        "Could not load the model files. Please check that "
        "model/spam_model.pkl and model/vectorizer.pkl exist."
    )
    st.exception(error)
    st.stop()

message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! You have won a free prize.",
    height=150
)

if st.button("🔍 Check Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        try:
            result, confidence = predict_message(
                message.strip(), model, vectorizer
            )

            st.subheader("Prediction Result")

            if result == "Spam":
                st.error("🚨 This message is classified as SPAM.")
            elif result == "Not Spam":
                st.success("✅ This message is classified as NOT SPAM.")
            else:
                st.info(f"Prediction: {result}")

            if confidence is not None:
                st.metric("Model confidence", f"{confidence:.2f}%")

        except Exception as error:
            st.error("Prediction failed. Please check the model and vectorizer.")
            st.exception(error)

st.divider()

st.subheader("💡 Try an Example")

if st.button("Example: Prize message"):
    st.session_state["example_message"] = (
        "Congratulations! You have won a free prize. Claim now!"
    )

if st.button("Example: Normal message"):
    st.session_state["example_message"] = (
        "Hi, are we meeting today after class?"
    )

if "example_message" in st.session_state:
    st.info(st.session_state["example_message"])
    st.caption("Copy this example into the message box above to test it.")
