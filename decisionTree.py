# By Jeff Yuanbo Han (u6617017), 2018-05-30.
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus

# Read in the csv file
attributes = {'class':2, 'cap-shape':6, 'cap-surface':4, 'cap-color':10, 'bruises?':2, 'odor':9,
              'gill-attachment':4, 'gill-spacing':3, 'gill-size':2, 'gill-color':12, 'stalk-shape':2,
              'stalk-root':7, 'stalk-surface-above-ring':4, 'stalk-surface-below-ring':4,
              'stalk-color-above-ring':9, 'stalk-color-below-ring':9, 'veil-type':2,
              'veil-color':4, 'ring-number':3, 'ring-type':8, 'spore-print-color':9, 'population':6,
              'habitat':7}

# Read original data (all nominally valued)
reader = pd.read_csv('Mushroom Data Set/agaricus-lepiota.data.csv', header=None, names=attributes.keys())
headers = reader.keys()

# Training data set
train = reader.as_matrix()[::]


def vec_feature(featuresNo, data, headers, get_feature_names=False):
    """
    Put required features into list of dict and list of class label.
    """
    featureList = []
    labelList = []
    for row in data:
        labelList.append(row[0])
        rowDict = {}
        for i in featuresNo:
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)

    # Vectorize features
    vec = DictVectorizer()
    dummyX = vec.fit_transform(featureList).toarray()

    # Vectorize class labels
    lb = preprocessing.LabelBinarizer()
    dummyY = lb.fit_transform(labelList)

    if get_feature_names:
        return dummyX, dummyY, vec.get_feature_names()
    else:
        return dummyX, dummyY


def model(X, Y, depth=2, vis=False):
    """
    Decision Tree Classification Model. Return the accuracy on itself.
    """
    dot_data = StringIO()
    clf = tree.DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    clf.fit(X, Y)
    accuracy = clf.score(X, Y)

    # Visualize model
    if vis:
        print(clf)
        print('Accuracy = {}%'.format(accuracy*100))
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("tree.pdf")

    return clf, accuracy


"""
# Predict
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX.reshape(1,-1))
print("predictedY: " + str(predictedY))
"""


if __name__ == '__main__':
    dummyX, dummyY = vec_feature(range(1, len(headers)), train, headers)
    model(dummyX, dummyY, depth=3, vis=True)
