import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load data
def load_data():
    news_df = pd.read_csv('train.csv')
    news_df = news_df.fillna(' ')
    news_df['content'] = news_df['author'] + ' ' + news_df['title']
    return news_df

news_df = load_data()

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Fit logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the preprocessed data and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
