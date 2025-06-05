import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def load_model():
    model_path = "./phishing_model"
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    return model, tokenizer

def predict_phishing(email_text, model, tokenizer):
    inputs = tokenizer(
        email_text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    is_phishing = bool(predicted_class)
    return {
        "is_phishing": is_phishing,
        "confidence": confidence,
        "prediction": "Phishing" if is_phishing else "Not Phishing"
    }

def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!")
    
    while True:
        print("\nEnter an email to check (or 'quit' to exit):")
        email = input("> ")
        
        if email.lower() == 'quit':
            break
            
        result = predict_phishing(email, model, tokenizer)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()
