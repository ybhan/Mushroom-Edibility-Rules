# By Jeff Yuanbo Han (u6617017), 2018-04-30.
import pandas as pd
import numpy as np

# All the attributes
attributes = {'class':2, 'cap-shape':6, 'cap-surface':4, 'cap-color':10, 'bruises?':2, 'odor':9,
              'gill-attachment':4, 'gill-spacing':3, 'gill-size':2, 'gill-color':12, 'stalk-shape':2,
              'stalk-root':7, 'stalk-surface-above-ring':4, 'stalk-surface-below-ring':4,
              'stalk-color-above-ring':9, 'stalk-color-below-ring':9, 'veil-type':2,
              'veil-color':4, 'ring-number':3, 'ring-type':8, 'spore-print-color':9, 'population':6,
              'habitat':7}

# Read original data (all nominally valued)
train = pd.read_csv('Mushroom Data Set/agaricus-lepiota.data.csv', header=None, names=attributes.keys())

# Data preprocessing (discretization)
train['cap-shape'] = train['cap-shape'].map(dict(zip('bcxfks', np.eye(6))))
train['cap-surface'] = train['cap-surface'].map(dict(zip('fgys', np.eye(4))))
train['cap-color'] = train['cap-color'].map(dict(zip('nbcgrpuewy', np.eye(10))))
train['bruises?'] = train['bruises?'].map(dict(zip('tf', np.eye(2))))
train['odor'] = train['odor'].map(dict(zip('alcyfmnps', np.eye(9))))
train['gill-attachment'] = train['gill-attachment'].map(dict(zip('adfn', np.eye(4))))
train['gill-spacing'] = train['gill-spacing'].map(dict(zip('cwd', np.eye(3))))
train['gill-size'] = train['gill-size'].map(dict(zip('bn', np.eye(2))))
train['gill-color'] = train['gill-color'].map(dict(zip('knbhgropuewy', np.eye(12))))
train['stalk-shape'] = train['stalk-shape'].map(dict(zip('et', np.eye(2))))
train['stalk-root'] = train['stalk-root'].map(dict(zip('bcuezr?', np.eye(7))))
train['stalk-surface-above-ring'] = train['stalk-surface-above-ring'].map(dict(zip('fyks', np.eye(4))))
train['stalk-surface-below-ring'] = train['stalk-surface-below-ring'].map(dict(zip('fyks', np.eye(4))))
train['stalk-color-above-ring'] = train['stalk-color-above-ring'].map(dict(zip('nbcgopewy', np.eye(9))))
train['stalk-color-below-ring'] = train['stalk-color-below-ring'].map(dict(zip('nbcgopewy', np.eye(9))))
train['veil-type'] = train['veil-type'].map(dict(zip('pu', np.eye(2))))
train['veil-color'] = train['veil-color'].map(dict(zip('nowy', np.eye(4))))
train['ring-number'] = train['ring-number'].map(dict(zip('not', np.eye(3))))
train['ring-type'] = train['ring-type'].map(dict(zip('ceflnpsz', np.eye(8))))
train['spore-print-color'] = train['spore-print-color'].map(dict(zip('knbhrouwy', np.eye(9))))
train['population'] = train['population'].map(dict(zip('acnsvy', np.eye(6))))
train['habitat'] = train['habitat'].map(dict(zip('glmpuwd', np.eye(7))))

train['class'] = train['class'].map({'e':1, 'p':0})

train = train.as_matrix()

train_data = np.empty(shape=[8124,127])
for i in range(np.size(train,0)):
    train_data[i,:] = np.hstack(train[i,:])

train_data[:,1:] = 2 * (train_data[:,1:] - 0.5)
