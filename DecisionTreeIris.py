import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus


# size of data set : 150
iris = load_iris()
test_idx = [0] #add indices of test data. 

#training data
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#train the classifiertrain_data = np.delete(iris.data,test_idx)
classifier = tree.DecisionTreeClassifier()  # Decision Tree classifier
classifier.fit(train_data,train_target)

#print prediction
print iris.target_names[test_target] #actual label
print iris.target_names[classifier.predict(test_data)]

dot_data = tree.export_graphviz(classifier, out_file=None,
	feature_names = iris.feature_names,
	class_names = iris.target_names,
	filled = True, rounded = True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_pdf("output_iris.pdf")
