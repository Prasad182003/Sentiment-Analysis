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