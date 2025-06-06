import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

@st.cache_resource
def load_model():
    model_path = "./phishing_model"
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def predict_phishing(email_text: str, model, tokenizer):
    inputs = tokenizer(
        email_text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

    is_phishing = bool(predicted_class)
    return {
        "is_phishing": is_phishing,
        "confidence": confidence,
        "prediction": "Phishing" if is_phishing else "Not Phishing"
    }

def main():
    st.set_page_config(page_title="Email Phishing Detector", page_icon="✉️")
    st.title("📧 Email Phishing Detector")
    st.markdown(
        """
        Paste the contents of an email below (or a snippet), then click **Predict** 
        to see if it’s likely phishing or not.
        """
    )

    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    st.success("Model loaded!")

    email_input = st.text_area("Enter email text here:", height=200)

    if st.button("Predict"):
        if not email_input.strip():
            st.warning("⚠️ Please paste some email text before clicking Predict.")
        else:
            with st.spinner("Running inference..."):
                result = predict_phishing(email_input, model, tokenizer)
            label = result["prediction"]
            conf_pct = result["confidence"] * 100

            if result["is_phishing"]:
                st.error(f"🏴 Prediction: **{label}**  \nConfidence: **{conf_pct:.2f}%**")
            else:
                st.success(f"✅ Prediction: **{label}**  \nConfidence: **{conf_pct:.2f}%**")

if __name__ == "__main__":
    main()
