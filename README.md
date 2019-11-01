## Two Approaches to Extract Logical Rules for Mushroom Edibility: Neural Networks and Genetic Algorithm ##

Project for ANU COMP4660/8420 (Bio-inspired Computing: Applications and Interfaces), Semester 1 2018.

By Yuanbo Han, 2018-05-31.

### Environment ###

- Python 3.6.3
  - numpy 1.14.3
  - matplotlib 2.2.2
  - pandas 0.22.0
  - torch 0.4.0
  - sklearn 0.19.1
  - pydotplus 2.0.2
- graphviz 2.40.1

Note that the above are just versions during experiment, not the least requirements.

### Data Set ###

*Mushroom Data Set/agaricus-lepiota.data.csv*

Original source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom)

### Codes ###

- `bpNN.py`
- `decisionTree.py`
- `displayWeight.py`
- `GATree.py`
- `load_data.py`

### BP Neural Networks ###

Run `bpNN.py`. It will read in the data, perform discretization, train a back-propagation neural network, and generate a file called "*net_weights*" which stores the weights in the model. To adjust parameters, see line 14\~26. To change the network structure, see line 29\~35.

Run `displayWeight.py`. It will read "*net_weights*" file and print the network weights for attribute values.

### GA + Decision Tree ###

Run `GATree.py`. It will read in the data, perform Genetic Algorithm for feature selection, and generate a "*tree.pdf*" which is the diagram of the final Decision Tree. Control parameters can be adjusted in line 6\~13.
