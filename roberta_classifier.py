# roberta_classifier.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class RobertaTwitterClassifier:
    def __init__(self):
        # Twitter-optimized RoBERTa model
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"

        # Initialize tokenizer with explicit max length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=512
        )

        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text):
        # Tokenize input with proper length management
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=512
        )

        # Model inference
        outputs = self.model(**inputs)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Return numpy array: [negative, neutral, positive] probabilities
        return probs.detach().numpy()

    @staticmethod
    def get_label_names():
        """Helper method to get human-readable labels"""
        return ["Negative", "Neutral", "Positive"]