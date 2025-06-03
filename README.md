# Sentiment Analysis Toolkit 🚀

A Python-based solution for classifying tweet sentiments using **Naive Bayes (NBC)**, improved NBC with feature selection (iNBC), and preprocessing pipelines. Perfect for learning NLP fundamentals or deploying lightweight sentiment analysis.

## Key Features ✨

- **Three Classifiers:**  
  - Standard Naive Bayes (NBC)  
  - Optimized NBC with chi-square feature selection (**iNBC**)
  - RoBERTa
- **Smart Preprocessing:**  
  - Handles hashtags, mentions, URLs, and slang (e.g., "loooove" → "love")  
  - Lemmatization + POS tagging for better word normalization  
- **Accuracy Boost:** iNBC achieves **82% accuracy** vs. NBC's 75%  

Code Structure 📂
├── main.py            # Main driver script
├── classifier.py       # Standard Naive Bayes
├── classifier1.py      # Optimized iNBC with chi-square
├── roberta_classifier.py      
├── sentiment_app.py    #Front-end
├── correcting.py # Text cleaning pipeline
├── tagAndLemmatize.py     # POS tagging + lemmatization
└── twitterData.csv       # Example dataset
