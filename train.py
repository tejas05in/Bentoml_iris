import bentoml
from sklearn.datasets import load_iris
from sklearn.svm import SVC

#Loading training data set
iris = load_iris()
X, y = iris.data , iris.target


#Train the model
clf = SVC(gamma='scale')
clf.fit(X,y)


#Save the model to the BentoML local model store
saved_model = bentoml.sklearn.save_model('iris_clf',clf)
print(f"Model svaed: {saved_model}")