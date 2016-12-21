import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,:2] #take two features
Y = iris.target

h = 0.01 #step size in mesh
logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X,Y)

# to visualize
x_min,x_max = X[:,0].min() - .5, X[:,0].max() + 0.5
y_min,y_max = X[:,1].min() - .5, X[:,1].max() + 0.5

xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
z = logreg.predict(np.c_[xx.ravel(),yy.ravel()])

z = z.reshape(xx.shape)

plt.figure(1,figsize = (4,4))
plt.pcolormesh(xx,yy,z)

#plot training points
plt.scatter(X[:,0],X[:,1],c = Y,edgecolors = 'k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()

print(iris.target_names)