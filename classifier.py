import pandas as pd
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from numpy.random import RandomState 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the CSV file.
data = pd.read_csv('spamham.csv', delimiter = ',\t', names = ['label', 'message'])

# Categorize "ham" as 0 and "spam" as 1.
data['label'] = data.label.map({'ham': 0, 'spam': 1})

corpus = []
ps = PorterStemmer()

# Cleaning the data.
for i in range(0, len(data)):
    # Handle regex and make all lowercase.
    message = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    message = message.lower()

    # Remove punctuation and stopwords.
    message = message.translate(str.maketrans('', '', string.punctuation))
    message = [ps.stem(word) for word in message.split() if not word in set(stopwords.words('english'))]

    message = ' '.join(message)

    corpus.append(message)


# Create a data frame from the processed data.
label = pd.DataFrame(data['label'])
message = pd.DataFrame(data['message'])

# Creating the Bag of Words model by converting the text data into vectors.
vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X = vectorizer.fit_transform(corpus).toarray()

y = pd.get_dummies(data['label'])
y = y.iloc[:, 1].values

# Splits the table into 80% training data and 20% testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = RandomState())

# Initialize multiple classification models.
svc = SVC(kernel = 'sigmoid', gamma = 1.0)
knc = KNeighborsClassifier(n_neighbors = 49)
mnb = MultinomialNB(alpha = 0.2)
dtc = DecisionTreeClassifier(min_samples_split = 7, random_state = 111)
lrc = LogisticRegression(solver = 'liblinear', penalty = 'l1')
rfc = RandomForestClassifier(n_estimators = 31, random_state = 111)

# Create a dictionary of variables and models.
clfs = {'SVC' : svc, 'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}

# Fit the data onto the models.
def train(clf, features, targets):
    clf.fit(features, targets)

def predict(clf, features):
    return (clf.predict(features))

# def show_auc(y_true, y_pred):
#     fpr, tpr, _ = roc_curve(y_true, y_pred)
#     roc_auc = auc(fpr, tpr)
#     plt.figure()
#     plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()

accuracyScore = []
aucScore = []

for k, v in clfs.items():
    train(v, X_train, y_train)
    pred = predict(v, X_test)

    acc = accuracy_score(y_test, pred)
    accuracyScore.append((k, acc))

    fpr, tpr, _ = metrics.roc_curve(y_test, pred)
    aucScore.append(auc(fpr, tpr))

    # show_auc(y_test, pred)
    
    print(k)
    
    print('Accuracy. Avg: %0.5f, Std: %0.5f' % (np.mean(accuracyScore), np.std(accuracyScore)))
    print('AUC. Avg: %0.5f, Std: %0.5f' % (np.mean(aucScore), np.std(aucScore)))

    cm = confusion_matrix(y_test, pred) # Checking classification results with confusion matrix.

    f, ax = plt.subplots(figsize = (5, 5))
    sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f", ax = ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()

# Model predictions

# Write functions to detect if the message is spam or not
def find(x):
    if x == 1:
        print("Message is SPAM")
    else:
        print("Message is NOT Spam")

# Replace text with any message of choice
newtext = ["Free entry"]
integers = vectorizer.transform(newtext)

# Naive Bayes
x = mnb.predict(integers)
find(x)
