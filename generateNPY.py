#
# Read data of two datasets and mix them accordingly
#
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

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
data = data.sample(frac=1, random_state=8416159)
data = data.sample(frac=1, random_state=56985)
data_np = data.to_numpy()

# size = data_np.shape[0]/20
# size = int(size)
# counter1 = 0
# counter0 = 0
# i = 0
# for x in range(0, size):
#     data_chunk = data_np[0 + 20 * i:20*(i+1)]
#     counter1 = 0
#     counter0 = 0
#     for d in data_chunk:
#         if d[1] == 0:
#             counter0 = counter0 + 1
#         if d[1] == 1:
#             counter1 = counter1 + 1
#     print("----------------------------------------")
#     print(counter0)
#     print(counter1)
#     i = i + 1

np.save("enron_processed_npy", data_np, allow_pickle=True)


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
data2 = data2.sample(frac=1, random_state=841776159)
data2 = data2.sample(frac=1, random_state=7822484)
data2_np = data2.to_numpy()
np.save("processed_data_processed_npy", data2_np, allow_pickle=True)

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


file_name = "n=" + str(n_drifts) + "_npy"
df_npy = df.to_numpy()
np.save(file_name, df_npy, allow_pickle=True)

size = df_npy.shape[0]/20
size = int(size)
counter1 = 0
counter0 = 0
i = 0
ohno = 0
for x in range(0, size):
    data_chunk = df_npy[0 + 20 * i:20*(i+1)]
    counter1 = 0
    counter0 = 0
    for d in data_chunk:
        if d[1] == 0:
            counter0 = counter0 + 1
        if d[1] == 1:
            counter1 = counter1 + 1
    print("----------------------------------------")
    if counter0 == 0:
        ohno = ohno+1

    if counter1 == 0:
        ohno = ohno+1
    print(counter0)
    print(counter1)
    i = i + 1
print('\n')
print(ohno)
print(len(df))