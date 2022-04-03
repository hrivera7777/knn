import pandas as pd
import numpy as np
import random
k = int(input("enter the k value: "))
# iris --> dataframe
iris = pd.read_csv("data/iris.csv", sep=';')

# save id and class
iris_classes = iris.iloc[:, [0] + [-1]]


# irisompare = iris.drop(columns=['class'], axis=1)


# # randomize data
iris = iris.sample(frac=1)

# separate into train and test data
train = iris.sample(frac=0.7, random_state=200)  # random state is a seed value
test = iris.drop(train.index)


def get_class(id_):
    """ based on an id returns a class """
    ids = iris_classes.loc[iris_classes['ID'] == id_]
    return ids["class"].values[0]


def get_clasification(arr):
    classes_count = {}
    for el in arr:
        train_class = get_class(el[1])
        if train_class not in classes_count.keys():
            classes_count[train_class] = 1
        else:
            classes_count[train_class] += 1
    # returns key with highest value
    return max(classes_count, key=classes_count.get)


def get_distance(arr_test, arr_train):
    """ get the distance between 2 arrays using the euclidean distance algorithm """

    # (ID,Longitud sepalo,ancho del sepalo,longitud del petalo,ancho del petalo)
    id_train = arr_train.pop(0)
    id_test = arr_test.pop(0)
    res = 0
    for el_test, el_train in zip(arr_test, arr_train):
        res += abs(el_test - el_train)
    return (id_test, id_train, res)


def get_nearest_neighbors(test, train, k):
    """
    returns a matrix with euclidean distance comparisons between 2 data sets
    k is the number of elements each ROW of the matrix HAS (distances)
    each element has the following structure [ID_test, ID_train, distance]
     """

    test = test.drop(columns=['class'], axis=1)
    train = train.drop(columns=['class'], axis=1)
    dmatrix = []
    for tt in test.iterrows():
        row = []
        d = 0
        for tr in train.iterrows():
            new_tt, new_tr = list(tt), list(tr)
            new_tt.pop(0)
            new_tr.pop(0)
            d = get_distance(list(new_tt[0].values), list(
                new_tr[0].values))  # (IDtest, ,IDtrain, dist)
            row.append(d)
        row = sorted(row, key=lambda t: t[2])[:k]
        dmatrix.append(row)
    return dmatrix


def validate_prediction(test_class, class_prediction):
    """ validates if a predicted class and the REAL class are EQUAL or NOT"""
    return test_class == class_prediction


def get_prediction(test, trains, k):
    # element --> [[[id_test, id_train, distance]], [...], [...]]
    dmatrix = get_nearest_neighbors(test, train, k)
    count = 0
    predictions = {}
    for arr in dmatrix:
        class_prediction = get_clasification(arr)
        predictions[arr[0][0]] = class_prediction
        if validate_prediction(get_class(arr[0][0]), class_prediction):
            count += 1

    return (f"accuracy: {round(count / len(test), 5) * 100}% 😊", f"prediction: {predictions}")


print(get_prediction(test, train, k))
