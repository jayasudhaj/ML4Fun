from sklearn.datasets import load_iris
from sklearn import naive_bayes
iris = load_iris()

#gaussian naive bayes. here likelihood is assumed to be gaussian
gnb = naive_bayes.GaussianNB()
classifier = gnb.fit(iris.data,iris.target)

prediction = classifier.predict(iris.data)
print("Predicted label of 1st sample "+iris.target_names[prediction[0]]);
print("Actual label of 1st sample "+iris.target_names[iris.target[0]])

#test for overfitting in NB
train_data = iris.data[1:100]
train_target = iris.target[1:100]
classifier = naive_bayes.GaussianNB().fit(train_data,train_target)
prediction = classifier.predict(train_data)
mispred = (prediction!=train_target).sum()
print(" Number of mislabelled samples in train data(100 samples): "+str(mispred))

test_data = iris.data[50:150]
test_target = iris.target[50:150]

prediction = classifier.predict(test_data)
mispred = (prediction!=test_target).sum()
print(" Number of mislabelled samples in test data: "+str(mispred))