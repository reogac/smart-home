#!/usr/bin/python

import modeling
import numpy as np
import pickle as pk

def print_confusion_matrix(c, labels):
    n = c.shape[0]
    s = "act\pred"
    for l in labels:
        s += "\t" + l
    print s
    for i in range(n):
        s=labels[i] + "\t"
        for j in range(n):
            s += "\t" + str(c[i][j])
        print s

def normalize(x):
    nrows, ncols = x.shape
    if nrows>1:
        for j in range(ncols-2):
            m = np.mean(x[:, j])
            std = np.std(x[:, j])
            if std>0.0:
                for i in range(nrows):
                    x[i, j] = (x[i,j]-m)/std
            else:
                for i in range(nrows):
                    x[i, j] = 0.0
    return x

def merge_data():

    data=[]
    on_nrows = 0
    off_nrows = 0

    for i in range(1,21):
        try:
            filename = "data/node" + str(i).zfill(2) + "_201508_cls.csv"
            print "process file: " + filename
            node_data = modeling.process_data(filename)
            if node_data:
                data.append([normalize(node_data[0]), node_data[1], normalize(node_data[2]), node_data[3]])
                on_nrows += node_data[0].shape[0]
                off_nrows += node_data[2].shape[0]
        except modeling.EngineError as e:
            pass


    #merge data

    x_on=np.empty([on_nrows, modeling.NUM_FEATURES], float)
    y_on=np.empty(on_nrows, int)
    x_off=np.empty([off_nrows, modeling.NUM_FEATURES], float)
    y_off=np.empty(off_nrows, int)

    on_nrows=0
    off_nrows=0
    for x_node_on, y_node_on, x_node_off, y_node_off in data:
        on_node_nrows = x_node_on.shape[0]

        x_on[on_nrows:on_nrows+on_node_nrows,:] = x_node_on
        y_on[on_nrows:on_nrows+on_node_nrows] = y_node_on
        on_nrows += on_node_nrows

        off_node_nrows = x_node_off.shape[0]
        x_off[off_nrows:off_nrows+off_node_nrows,:] = x_node_off
        y_off[off_nrows:off_nrows+off_node_nrows] = y_node_off
        off_nrows += off_node_nrows
    data = [x_on, y_on, x_off, y_off]
    pk.dump(data, open("merged-data.pk", "wb"))
    return data


try:
    data = pk.load(open("merged-data.pk", "rb"))
except Exception as e:
    print e
    data = merge_data()
    pass


#cross validation evaluation

print "\nPredict TURN ON"

con_mats =  modeling.evaluate_model(data[2], data[3], "forest")
fold = 1
for c1, c2 in con_mats:
    print "\nPrediction performance for fold " + str(fold)
    print "\n... on training data"
    print_confusion_matrix(c1, ("NOTHING", "TURN-ON"))
    print "\n... on testing data"
    print_confusion_matrix(c2, ("NOTHING", "TURN-ON"))
    fold += 1
    print  "\n\nPerformance for TURN OFF prediction"


print "\nPredict TURN OFF "

con_mats =  modeling.evaluate_model(data[0], data[1], "forest")

fold = 1
for c1, c2 in con_mats:
    print "\nPrediction performance for fold " + str(fold)
    print "\n... on training data"
    print_confusion_matrix(c1, ("DO-NOTHING", "TURN-OFF"))
    print "\n... on testing data"
    print_confusion_matrix(c2, ("DO-NOTHING", "TURN-OFF"))
    fold += 1

