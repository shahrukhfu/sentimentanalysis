# Sentiment Analysis Toolkit 🚀

A Python-based solution for classifying tweet sentiments using **Naive Bayes (NBC)**, improved NBC with feature selection (iNBC), and preprocessing pipelines. Perfect for learning NLP fundamentals or deploying lightweight sentiment analysis.


Why This Project?
Airline companies receive thousands of tweets and reviews daily. This tool helps analyze customer sentiment by:

- Analyzing text (tweets, reviews, surveys)
- Classifying sentiment (Positive/Neutral/Negative)
- Visualizing results for easy interpretation
- Offering two models: Fast Naive Bayes and high-accuracy RoBERTa

Built for customer service teams, social media managers, or anyone analyzing public opinion about airlines.

Tech Stack
Component        Tools Used               Purpose
Text Cleaning    NLTK, Regex              Remove hashtags, URLs, emojis
Tagging          NLTK's Treebank          Word normalization
Models           Naive Bayes (NBC/iNBC) + RoBERTa   Balance speed and accuracy
Frontend         Streamlit                Interactive UI

Quick Start
1. Install dependencies:
git clone https://github.com/yourusername/airline-sentiment.git  
cd airline-sentiment  
pip install -r requirements.txt  
python -m nltk.downloader punkt wordnet averaged_perceptron_tagger stopwords treebank  

2. Run the application:
streamlit run sentiment_app.py  
The app will open in your browser at localhost:8501

3. Usage:
- Paste airline tweets/reviews
- Click Analyze
- View sentiment analysis results and visualizations

Files Explained
File                   Purpose
sentiment_app.py       Main application (UI and model integration)
correcting.py          Text cleaning (removes URLs, slang, etc.)
classifier.py          Standard Naive Bayes model
classifier2.py         Improved Naive Bayes (iNBC)
roberta_classifier.py  RoBERTa implementation (optimized for Twitter)
tagAndLemmatize.py     Handles word tagging/lemmatization

Use Cases
- Airlines: Monitor tweet sentiment in real-time
- Travel Agencies: Compare customer satisfaction across airlines
- Researchers: Study emotional trends in travel feedback

How to Contribute
- Report bugs
- Suggest new features
- Improve documentation

First time contributors should check out CONTRIBUTING.md (coming soon).

License
MIT License - Free to use, modify, and share.
