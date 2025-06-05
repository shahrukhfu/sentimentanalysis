# Main File with Fixed Prediction Flow
import nltk
import os
from correcting import ExtractingData
from classifier import ClassifyData
import classifier2
from tagAndLemmatize import TaggingAndLemmatize
from roberta_classifier import RobertaTwitterClassifier


class ClassificationOfData:
    def __init__(self):
        # Configure NLTK data path
        nltk_dir = (os.path.
                    join(os.path.expanduser("~"), "nltk_data"))
        if not os.path.exists(nltk_dir):
            raise FileNotFoundError(f"Run nltk_setup.py first! Missing: {nltk_dir}")
        nltk.data.path.append(nltk_dir)

    def run(self):
        # Load training data
        try:
            location = input('Enter training file name: ')
            with open(location, encoding="utf8") as file:
                lines = file.readlines()
            sents, labelsData = self.correctAndReplaceData(lines)
        except Exception as e:
            print(f"Error loading file: {e}")
            return

        # Model selection
        while True:
            option = input('\nChoose model [NBC/iNBC/RoBERTa]: ').strip().lower()
            if option in ('nbc', 'inbc', 'roberta'):
                break
            print("Invalid option! Please choose NBC, iNBC, or RoBERTa")

        # Initialize classifier
        classifier = self.classifierNBCData(sents, labelsData, option)

        # Prediction loop
        while True:
            user_input = input('\nEnter text to analyze (or "exit" to quit): ').strip()

            if user_input.lower() in ('exit', 'quit', 'q'):
                print("\nExiting...")
                break

            if not user_input:
                print("⚠ Please enter some text!")
                continue

            self.process_and_predict(classifier, user_input, option)

    def process_and_predict(self, classifier, text, option):
        try:
            # Preprocess
            processed_text = ExtractingData().runMethods([text])
            if not processed_text or not processed_text[0].strip():
                print("⚠ Text became empty after preprocessing!")
                return

            lemmatized_text = TaggingAndLemmatize().LemmatizeSents(processed_text)

            # Predict
            for clean_text in lemmatized_text:
                if not clean_text.strip():
                    continue

                if option == 'roberta':
                    self.handle_roberta_prediction(classifier, clean_text)
                else:
                    self.handle_naive_bayes_prediction(classifier, clean_text, option)
        except Exception as e:
            print(f"Error during processing: {e}")

    def handle_roberta_prediction(self, classifier, text):
        try:
            probs = classifier.predict(text)[0]
            print(f"\n🐦 RoBERTa Twitter Analysis:")
            print(f"  Negative: {probs[0]:.2%}")
            print(f"  Neutral:  {probs[1]:.2%}")
            print(f"  Positive: {probs[2]:.2%}")
        except Exception as e:
            print(f"Prediction error: {e}")

    def handle_naive_bayes_prediction(self, classifier, text, option):
        try:
            tokens = nltk.tokenize.word_tokenize(text)
            features = (classifier2.wordDict(tokens) if option == 'inbc'
                        else ClassifyData().wordDict(tokens))
            label = classifier.classify(features)
            print(f"\n📊 Naive Bayes Prediction: {label}")
        except Exception as e:
            print(f"Prediction error: {e}")

    def correctAndReplaceData(self, file):
        sents = []
        labelsData = []
        try:
            numb = int(input('Enter number of records to read: '))
            print('Processing training data...')
            for a in file[:numb]:
                if len(a.split(",")) == 3:
                    labelsData.append(a.split(",")[0])
                    sents.append(a.split(",")[2])
            sentsED = ExtractingData().runMethods(sents)
            sentsTAL = TaggingAndLemmatize().LemmatizeSents(sentsED)
            return sentsTAL, labelsData
        except Exception as e:
            print(f"Data processing error: {e}")
            return [], []

    def classifierNBCData(self, sents, labelsData, option):
        try:
            if option == 'inbc':
                print('\n🚀 Starting Improved Naive Bayes...')
                return classifier2.implementMethods(sents, labelsData, 'a')
            elif option == 'roberta':
                print('\n🤖 Initializing RoBERTa Twitter model...')
                return RobertaTwitterClassifier()
            else:
                print('\n🔍 Starting Standard Naive Bayes...')
                return ClassifyData().implementMethods(sents, labelsData, 'a')
        except Exception as e:
            print(f"Model initialization failed: {e}")
            return None


if __name__ == "__main__":
    ClassificationOfData().run()