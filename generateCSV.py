#
# Read data of two datasets and mix them accordingly
#
from sklearn.preprocessing import LabelEncoder
import pandas as pd

le = LabelEncoder()
data = pd.read_csv("enron_spam_data.csv")
data = data.fillna('')
data["Messages"] = data["Subject"] + ' ' + data["Message"]
data = data.drop("Message", axis=1)
data = data.drop("Message ID", axis=1)
data = data.drop("Date", axis=1)
data = data.drop("Subject", axis=1)

data['Messages'] = data['Messages'].str.lower()
data["Messages"] = [str(x).replace(':', ' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace(',', ' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace('.', ' ') for x in data["Messages"]]
data["Messages"] = [str(x).replace('-', ' ') for x in data["Messages"]]

data["Spam/Ham"] = le.fit_transform(data["Spam/Ham"])
data = data[["Messages", "Spam/Ham"]]
data.to_csv("enron_processed.csv", index=False, header=False)

#
#              _._     _,-'""`-._
#             (,-.`._,'(       |\`-/|
#                  `-.-' \ )-`( , o o)
#                       `-    \`_`"'-
#

data2 = pd.read_csv("processed_data.csv")
data2 = data2.fillna('')
data2["Messages"] = data2["subject"] + ' ' + data2["message"]
data2 = data2.drop("message", axis=1)
data2 = data2.drop("email_from", axis=1)
data2 = data2.drop("email_to", axis=1)
data2 = data2.drop("subject", axis=1)

data2['Messages'] = data2['Messages'].str.lower()
data2["Messages"] = [str(x).replace(':', ' ') for x in data2["Messages"]]
data2["Messages"] = [str(x).replace(',', ' ') for x in data2["Messages"]]
data2["Messages"] = [str(x).replace('.', ' ') for x in data2["Messages"]]
data2["Messages"] = [str(x).replace('-', ' ') for x in data2["Messages"]]

data2 = data2[["Messages", "label"]]
data2.rename(columns={'label': 'Spam/Ham'}, inplace=True)
data2.to_csv("processed_data_processed.csv", index=False, header=False)

# mix two datasets
n_drifts = 3
stop1 = int(len(data) / n_drifts)
stop2 = int(len(data2) / n_drifts)
iter1 = 0
iter2 = 0
df = pd.DataFrame(columns=["Messages", "Spam/Ham"])
while iter1 < n_drifts and iter2 < n_drifts:
    df = df.append(data[iter1 * stop1:stop1 * (iter1 + 1)])
    iter1 += 1
    df = df.append(data2[iter2 * stop2:stop2 * (iter2 + 1)])
    iter2 += 1

modulo1 = len(data) % n_drifts
modulo2 = len(data2) % n_drifts
if modulo1 != 0:
    df = df.append(data[stop1 * n_drifts : len(data) + 1])

if modulo2 != 0:
    df = df.append(data2[stop2 * n_drifts : len(data2) + 1])


file_name = "n=" + str(n_drifts) + ".csv"
df.to_csv(file_name, index=False, header=False)

print(len(df))