import zipfile
import pandas as pd
from textblob import TextBlob
import textstat


df = pd.read_csv("C:\\Mental-Health-Twitter.csv", encoding='latin1')

df.info()

cf = df.copy()

cf['post_created'] = cf['post_created'].astype('category')
cf['post_text'] = cf['post_text'].astype('category')

cf.info()

# Create new columns for sentiment
cf['polarity'] = cf['post_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
cf['subjectivity'] = cf['post_text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

print(cf.head())

# Sentiment label (categorical)
def get_sentiment_label(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

cf['sentiment_label'] = cf['polarity'].apply(get_sentiment_label)

# Readability metrics from textstat
cf['flesch_ease'] = cf['post_text'].apply(lambda x: textstat.flesch_reading_ease(str(x)))
cf['flesch_grade'] = cf['post_text'].apply(lambda x: textstat.flesch_kincaid_grade(str(x)))
cf['gunning_fog'] = cf['post_text'].apply(lambda x: textstat.gunning_fog(str(x)))
cf['smog_index'] = cf['post_text'].apply(lambda x: textstat.smog_index(str(x)))
cf['automated_index'] = cf['post_text'].apply(lambda x: textstat.automated_readability_index(str(x)))

# Optional: overall readability category (simpler for model understanding)
def interpret_flesch(score):
    if score >= 90:
        return 'Very Easy'
    elif score >= 80:
        return 'Easy'
    elif score >= 70:
        return 'Fairly Easy'
    elif score >= 60:
        return 'Standard'
    elif score >= 50:
        return 'Fairly Difficult'
    elif score >= 30:
        return 'Difficult'
    else:
        return 'Very Difficult'

cf['readability_level'] = cf['flesch_ease'].apply(interpret_flesch)


print(cf.head())
