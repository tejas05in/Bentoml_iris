import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service('iris_classifier', runners=[iris_clf_runner]) #any number of models can be specified in list
#Eg. [onehotencoder, standardscaler, classifier]

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series:np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result

#to run this file use: bentoml serve service.py:svc --reload