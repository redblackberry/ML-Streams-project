import matplotlib.pyplot as plt
import strlearn as sl
from sklearn.naive_bayes import GaussianNB
from strlearn.utils import scores_to_cummean
from AWE_StreamLearn import AWE_Main
import numpy as np
from sklearn.metrics import accuracy_score
from strlearn.metrics import geometric_mean_score_1
from tabulate import tabulate
from sklearn.metrics import balanced_accuracy_score, f1_score
import xgboost as xgb
from strlearn.ensembles import WAE, SEA

stream = sl.streams.StreamGenerator(n_chunks=100,
                                    chunk_size=20,
                                    n_classes=2,
                                    n_drifts=1,
                                    n_features=10)

# print(stream.get_chunk())

stream.save_to_npy('test.npy')

stream = sl.streams.NPYParser('test.npy', chunk_size=20, n_chunks=100)
# print(stream.get_chunk())


clfs = [
    # AWE_Main(GaussianNB(), n_estimators=10),
    SEA(GaussianNB(),n_estimators=10),
    WAE(GaussianNB(),n_estimators=10),
]
clf_names = [
    # "AWE",
    "SEA",
    "WAE"
]

# Wybrana metryka
# metrics = [accuracy_score,
#            precision]

metrics = [balanced_accuracy_score,
           geometric_mean_score_1,
           f1_score]

# Nazwy metryk
metrics_names = ["Bal acc score",
                 "G-Mean",
                 "F1 Score"]

evaluator = sl.evaluators.TestThenTrain(metrics)
evaluator.process(stream,clfs)
srednia = []
odchylenie = []
fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
fig, ax2 = plt.subplots(1, len(metrics), figsize=(24, 8))

scores_cm = scores_to_cummean(evaluator.scores)
print(scores_cm.shape)
for m, metric in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax2[m].set_title(metrics_names[m])
    ax[m].set_ylim(0, 1)
    ax2[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(scores_cm[i,:,m], label=clf_names[i])
        ax2[m].plot(evaluator.scores[i,:,m], label=clf_names[i])
        srednia.append(np.mean(evaluator.scores[i, :, m]))
        odchylenie.append(np.std(evaluator.scores[i, :, m]))

    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
    ax2[m].legend()

print("AWE")
print ("Balanced: ",srednia[0])
print ("G-Mean",srednia[1])
print ("F1 Score",srednia[2])
print ("SEA")
print ("Balanced: ",srednia[3])
print ("G-Mean",srednia[4])
print ("F1 Score",srednia[5])
print ("WAE")
print ("Balanced: ",srednia[6])
print ("G-Mean",srednia[7])
print ("F1 Score",srednia[8])

# table1 = [[metrics_names[0],'AWE','ADA','HTC'],
#          ["Srednia",(format(srednia[0],".3f")),(format(srednia[1],".3f")),(format(srednia[2],".3f"))],
#          ["Odchylenie",(format(odchylenie[0],".3f")),(format(odchylenie[1],".3f")),(format(odchylenie[2],".3f"))]]
# table2 = [[metrics_names[1],'AWE','ADA','HTC'],
#          ["Srednia",(format(srednia[3],".3f")),(format(srednia[4],".3f")),(format(srednia[5],".3f"))],
#          ["Odchylenie",(format(odchylenie[3],".3f")),(format(odchylenie[4],".3f")),(format(odchylenie[5],".3f"))]]
# print(tabulate(table1))
# print(tabulate(table2))
plt.show()