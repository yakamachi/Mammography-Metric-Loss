import sys

import cv2
import matplotlib.pyplot as plt
import io
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from sklearn import metrics

def normalize_image(image):

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grey

def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image, cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9,9))

    axs = fig.subplots(1, 3)
    show(axs[0], anchor)
    show(axs[1], positive)
    show(axs[2], negative)

    plt.show()

def save_result(model, img, labels, dataset_name, suffix, class_count):

    out_m = io.open('./TSV/meta_'+ dataset_name + '_' + suffix + '.tsv', 'w', encoding='utf-8')
    tmp = model.predict(img)
    result_class = tmp[1]
    result_emb = tmp[0]
    [out_m.write(str(x) + "\n") for x in labels]
    out_m.close()

    list = []
    classes = zip(labels, result_class)
    for real, pred in classes:
        class_pred = np.argmax(pred)
        p1 = pred[0]
        p2 = pred[1]
        p3 = 0
        if class_count == 3:
            p3 = pred[2]
        p = class_pred
        r = real
        r1 = 0
        r2 = 0
        r3 = 0
        if (r == 0):
            r1 = 1
        elif (r == 1):
            r2 = 1
        else:
            r3 = 1
        if class_count == 3:
            list.append([p1, p2, p3, p, r, r1, r2, r3])
        else:
            list.append([p1, p2, p, r, r1, r2])

    df = []
    if class_count == 3:
        df = pd.DataFrame(list, columns=["Class1 Prob Pred", "Class2 Prob Pred", "Class3 Prob Pred", "Predicted", "Real", "Class1 Prob Real", "Class2 Prob Real", "Class3 Prob Real"])
    else:
        df = pd.DataFrame(list,
                          columns=["Class1 Prob Pred", "Class2 Prob Pred", "Predicted", "Real", "Class1 Prob Real", "Class2 Prob Real"])
    df.to_csv('./CSV/prediction_' + dataset_name + '_' + suffix + '.csv', index=False)

    np.savetxt("./tsv/vecs_emb_" + dataset_name+ "_" + suffix +".tsv", result_emb, delimiter='\t')
    np.savetxt("./tsv/vecs_class_" + dataset_name + "_" + suffix + ".tsv", result_class, delimiter='\t')

    return [result_emb, result_class]

def save_result_class(model, img, labels, dataset_name, suffix, class_count):

    #out_m = io.open('./TSV/meta_'+ dataset_name + '_' + suffix + '.tsv', 'w', encoding='utf-8')
    tmp = model.predict(img)
    result_class = tmp
    #result_emb = tmp[0]
    #[out_m.write(str(x) + "\n") for x in labels]
    #out_m.close()

    list = []
    classes = zip(labels, result_class)
    for real, pred in classes:
        class_pred = np.argmax(pred)
        p1 = pred[0]
        p2 = pred[1]
        p3 = 0
        if class_count == 3:
            p3 = pred[2]
        p = class_pred
        r = real
        r1 = 0
        r2 = 0
        r3 = 0
        if (r == 0):
            r1 = 1
        elif (r == 1):
            r2 = 1
        else:
            r3 = 1
        if class_count == 3:
            list.append([p1, p2, p3, p, r, r1, r2, r3])
        else:
            list.append([p1, p2, p, r, r1, r2])

    df = []
    if class_count == 3:
        df = pd.DataFrame(list, columns=["Class1 Prob Pred", "Class2 Prob Pred", "Class3 Prob Pred", "Predicted", "Real", "Class1 Prob Real", "Class2 Prob Real", "Class3 Prob Real"])
    else:
        df = pd.DataFrame(list,
                          columns=["Class1 Prob Pred", "Class2 Prob Pred", "Predicted", "Real", "Class1 Prob Real", "Class2 Prob Real"])
    df.to_csv('./CSV/prediction_' + dataset_name + '_' + suffix + '.csv', index=False)

def save_results_knn(labels, result_class, class_count, dataset_name):
    list = []
    classes = zip(labels, result_class)
    for real, pred in classes:
        class_pred = np.argmax(pred)
        p1 = pred[0]
        p2 = pred[1]
        p3 = 0
        if class_count == 3:
            p3 = pred[2]
        p = class_pred
        r = real
        r1 = 0
        r2 = 0
        r3 = 0
        if (r == 0):
            r1 = 1
        elif (r == 1):
            r2 = 1
        else:
            r3 = 1
        if class_count == 3:
            list.append([p1, p2, p3, p, r, r1, r2, r3])
        else:
            list.append([p1, p2, p, r, r1, r2])

    df = []
    if class_count == 3:
        df = pd.DataFrame(list,
                          columns=["Class1 Prob Pred", "Class2 Prob Pred", "Class3 Prob Pred", "Predicted", "Real",
                                   "Class1 Prob Real", "Class2 Prob Real", "Class3 Prob Real"])
    else:
        df = pd.DataFrame(list,
                          columns=["Class1 Prob Pred", "Class2 Prob Pred", "Predicted", "Real", "Class1 Prob Real",
                                   "Class2 Prob Real"])
    df.to_csv('./CSV/prediction_' + dataset_name + '_with_knn.csv', index=False)

def count_statistics(file_name):
    df = pd.read_csv('./CSV/'+file_name)

    array = df.to_numpy()

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    all = 0
    correct = 0

    p = []
    r = []

    for row in array:
        pred = row[2]
        real = row[3]

        if (pred == 0 and pred == real):
            TN += 1
            correct += 1
        elif ((pred == 1 or pred == 2) and (real == 1 or real == 2)):
            TP += 1
            correct += 1
        elif (pred == 0 and pred != real):
            FN += 1
        elif ((pred == 1 or pred == 2) and (real == 0)):
            FP += 1

        all += 1

    accuracy = correct / all
    sumTPFN = TP+FN
    if(sumTPFN == 0):
        sumTPFN = sys.maxsize
    sensitivity = TP / (sumTPFN)
    sumTNFP = TN + FP
    if (sumTNFP == 0):
        sumTNFP = sys.maxsize
    specificity = TN / (sumTNFP)
    sumTPFP = TP + FP
    if (sumTPFP == 0):
        sumTPFP = sys.maxsize
    precision = TP / (sumTPFP)
    fallout = FP / (sumTNFP)

    p1 = array[:,0]
    p2 = array[:,1]
    r1 = array[:,4]
    r2 = array[:,5]

    p = array[:,2]
    r = array[:,3]

    fpr, tpr, thr = metrics.roc_curve(r, p, pos_label=1)
    AUC = metrics.auc(fpr, tpr)


    return [file_name, accuracy, sensitivity, specificity, precision, fallout, AUC]



