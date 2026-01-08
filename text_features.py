import pandas as pd
import re
from textblob import TextBlob
import textstat
from sklearn.base import BaseEstimator, TransformerMixin

class TextFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, text_col="post_text"):
        self.text_col = text_col
        self.absolutist_words = {'always', 'never', 'completely', 'totally', 'entirely', 'forever', 'noone', 'nobody'}
        # General risk words
        self.depressive_lexicon = {'sad', 'hopeless', 'worthless', 'tired', 'empty', 'alone', 'dark', 'pain', 'hurt', 'miserable', 'exhausted'}
        # Extreme/Emergency indicators
        self.emergency_lexicon = {'die', 'suicide', 'kill', 'end it', 'death', 'goodbye', 'kill myself', 'wanna die'}

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
        text = re.sub(r'@\w+', '', text) 
        text = re.sub(r'#', '', text) 
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.text_col] = X[self.text_col].apply(self.clean_text)
        text = X[self.text_col]

        # Sentiment
        polarity = text.apply(lambda t: TextBlob(t).sentiment.polarity)
        
        # Risk word extraction
        def find_words(t, lexicon):
            return [w for w in lexicon if w in t] # Simple substring match for better detection
        
        emergency_found = text.apply(lambda t: find_words(t, self.emergency_lexicon))
        risk_found = text.apply(lambda t: find_words(t, self.depressive_lexicon))
        
        return pd.DataFrame({
            "post_text": text,
            "polarity": polarity,
            "emergency_words": emergency_found,
            "risk_words": risk_found,
            "i_usage": text.apply(lambda t: len(re.findall(r'\b(i|me|my|myself|mine)\b', t))),
            # Keep these for the XGBoost model pipeline compatibility
            "subjectivity": text.apply(lambda t: TextBlob(t).sentiment.subjectivity),
            "abs_usage": text.apply(lambda t: len([w for w in t.split() if w in self.absolutist_words])),
            "dep_word_count": risk_found.apply(len),
            "flesch_ease": text.apply(lambda t: textstat.flesch_reading_ease(t)),
            "sentiment_label": polarity.apply(lambda p: "Positive" if p > 0.1 else ("Neutral" if p >= -0.1 else "Negative"))
        })