import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
################################
desired_width = 42069
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
################################

le = LabelEncoder()
data = pd.read_csv("enron_spam_data.csv")
# print(data.head)
data["Messages"] = data["Subject"] + data["Message"]
data = data.drop("Message", axis=1)
data = data.drop("Message ID", axis=1) #useless
# print(data.head)

#male literki wszedzie, ale chyba i tak wszedzie sa male, ale na wszelki wypadek
data['Messages'] = data['Messages'].str.lower()
data['Subject'] = data['Subject'].str.lower()

#usuwanie interpunkcji, inaczej nie umiem XDDDDD, bo nie dziala
data["Messages"] = [str(x).replace(':', ' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace(',', ' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace('.', ' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace('-', ' ') for x in data["Messages"]]
data["Subject"] = [str(x).replace(':', ' ') for x in data["Subject"]]
data["Subject"] = [str(x).replace(',', ' ') for x in data["Subject"]]
data["Subject"] = [str(x).replace('.', ' ') for x in data["Subject"]]
# print(data.head)

data["Spam/Ham"] = le.fit_transform(data["Spam/Ham"])
print(data.head)

Text = []
for line in data["Messages"]:
    words = line.split(" ")
    Text.append(words)

X_train, X_test , y_train, y_test = train_test_split(data["Messages"], data["Spam/Ham"] , test_size=0.5)
# Vectorizer = CountVectorizer()
Vectorizer = TfidfVectorizer()
count= Vectorizer.fit_transform(X_train.values)
Spam_detection = MultinomialNB()
targets = y_train.values
Spam_detection.fit(count, targets)
y_predict = Spam_detection.predict(Vectorizer.transform(X_test))
print(accuracy_score(y_test, y_predict))

cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(classification_report(y_test , y_predict))