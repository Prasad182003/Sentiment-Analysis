!pip install nltk
!pip install twython
!pip install textblob
!pip install wordcloud

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("amazon_reviews.csv")

df.head()

df['reviewText']

df['reviewText'] = df['reviewText'].str.lower()

df['reviewText']

df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '', regex=True)

df['reviewText']

df['reviewText'] = df['reviewText'].str.replace('\d', '', regex=True)

df['reviewText']

nltk.download('stopwords')

sw = stopwords.words('english')

sw

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

df['reviewText']

temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()

temp_df

drops = temp_df[temp_df <= 1]

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

df['reviewText']

temp_df_1 = pd.Series(' '.join(df['reviewText']).split()).value_counts()

temp_df_1

"""# New Section

token process
"""

import nltk
nltk.download('punkt')

df["reviewText"].apply(lambda x: TextBlob(x).words).head()

""" Lemmatization

"""

nltk.download('wordnet')
nltk.download('omw-1.4')

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df['reviewText']

"""term frequency calculation"""

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ['words', 'tf']

tf_sorted = tf.sort_values(by='tf', ascending=False)

tf_sorted

from matplotlib import pyplot as plt
tf_sorted['tf'].plot(kind='hist', bins=20, title='tf')
plt.gca().spines[['top', 'right',]].set_visible(False)

"""barplot"""

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show(block=True)

"""word cloud"""

text = " ".join(i for i in df.reviewText)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Sentiment Analysis

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

sia.polarity_scores("The film was awesome")

#sia.polarity_scores("Very nice performance Only thing that is missing in kit is cable wire battery for remote AC Copper wire tape Installation charges were least  plus gst  However they add accessories like outdoor unit angle stand tape power cable  core drainage pipe tape installation charges separately though its mandatory work They all add up to ")

sia.polarity_scores("I liked this music but it is not good as the other one")

sia.polarity_scores("I have it in my phone and it never skips a beat. File transfers are speedy and have not had any corruption issues or memory fade issues as I would expect from the Sandisk brand. Great card to own. Why entrust your precious files to a slightly cheaper piece of crap? If you lose everything can you forgive yourself for not spending the extra couple bucks on a trusted product that goes through good QA?")

sia.polarity_scores("THE NAME OF ITSELF SPEAKS OUT. GO SANDISK GO!")

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])

df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviewText"]

df.head(40)