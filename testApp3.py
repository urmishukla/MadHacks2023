import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk import tokenize
from sklearn import metrics
import itertools
from sklearn.linear_model import LogisticRegression
#import wordcloud
#from wordcloud import WordCloud
from sklearn import metrics
import itertools
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

fakeSerious = pd.read_csv("PolitiFact_fake_news_content.csv")
realSerious = pd.read_csv("PolitiFact_real_news_content.csv")

#fakeSerious = pd.read_csv("Fake.csv")
#realSerious = pd.read_csv("True.csv")
#print(fakeSerious.shape)

# create column to identify real and fake news
fakeSerious['target'] = 'fake'
realSerious['target'] = 'real'

# combine both datasets into one large dataset
data = pd.concat([fakeSerious, realSerious]).reset_index(drop= True)

# shuffle to avoid data contamination
data = shuffle(data)
data = data.reset_index(drop=True)

# removing unnecessary fields, applying lowercase to avoid case sensitivity
data.drop(["publish_date"], axis=1, inplace=True)
data.drop(["title"], axis=1, inplace=True)
data.drop(["id"], axis=1, inplace=True)
data.drop(["url"], axis=1, inplace=True)
data.drop(["top_img"], axis=1, inplace=True)
data.drop(["movies"], axis=1, inplace=True)
data.drop(["meta_data"], axis=1, inplace=True)
data.drop(["canonical_link"], axis=1, inplace=True)
data.drop(["images"], axis=1, inplace=True)
data['text'] = data['text'].apply(lambda x: x.lower())

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#print(data.head())

# fake data most commonly used words
fake_data = data[data["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])


# real
real_data = data[data["target"] == "true"]
all_words = ' '.join([text for text in fake_data.text])


token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()


# most common words in fake news
#counter(data[data["target"] == "fake"], "text", 20)

# most common words in real news
#counter(data[data["target"] == "real"], "text", 20)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


X_train, X_test, y_train, y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)

# Vectorizing and applying TF-IDF

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

# Fitting the model
model = pipe.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])


# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
# Fitting the model
model = pipe.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])