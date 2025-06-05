# sentiment_app.py
import streamlit as st
import pandas as pd
import nltk
from correcting import ExtractingData
from tagAndLemmatize import TaggingAndLemmatize
from roberta_classifier import RobertaTwitterClassifier

# Configure NLTK
nltk.data.path.append("nltk_data")


class SentimentAnalyzer:
    def __init__(self):
        self.extractor = ExtractingData()
        self.lemmatizer = TaggingAndLemmatize()
        self.classifier = RobertaTwitterClassifier()  # Only RoBERTa

    def predict(self, text):
        """Make prediction with RoBERTa"""
        processed = self.extractor.runMethods([text])
        if not processed or not processed[0].strip():
            return None
        clean_text = self.lemmatizer.LemmatizeSents(processed)[0]
        return self.classifier.predict(clean_text)[0]


def main():
    st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SentimentAnalyzer()

    st.title("Twitter Sentiment Analysis")
    st.markdown("Analyze sentiment using RoBERTa (Twitter-optimized model)")

    # Text input
    text_input = st.text_area(
        "Enter tweet or text to analyze:",
        height=150,
        placeholder="Paste your text here..."
    )

    # Sample tweets
    st.write("Try these examples:")
    samples = [
        "This flight was amazing! Great service",
        "Worst experience ever, would never fly again",
        "It was okay, nothing special"
    ]

    cols = st.columns(len(samples))
    for i, sample in enumerate(samples):
        if cols[i].button(sample):
            text_input = sample

    # Prediction
    if st.button("Analyze Sentiment"):
        if not text_input.strip():
            st.warning("Please enter some text!")
            st.stop()

        with st.spinner("Analyzing with RoBERTa..."):
            result = st.session_state.analyzer.predict(text_input)

            if result is None:
                st.error("Analysis failed - text became empty after preprocessing")
                st.stop()

            st.subheader("Sentiment Probabilities")
            cols = st.columns(3)
            cols[0].metric("Negative", f"{result[0]:.1%}", delta_color="inverse")
            cols[1].metric("Neutral", f"{result[1]:.1%}")
            cols[2].metric("Positive", f"{result[2]:.1%}")

            # Visualization
            chart_data = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Score': result
            })
            st.bar_chart(chart_data.set_index('Sentiment'), use_container_width=True)

            # Show processing steps
            with st.expander("Show processing steps"):
                processed = st.session_state.analyzer.extractor.runMethods([text_input])[0]
                lemmatized = st.session_state.analyzer.lemmatizer.LemmatizeSents([processed])[0]
                st.write("Original:", text_input)
                st.write("After cleaning:", processed)
                st.write("After lemmatization:", lemmatized)


if __name__ == "__main__":
    main()