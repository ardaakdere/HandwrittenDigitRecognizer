import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import os

path = os.path.dirname(__file__)
os.chdir(path)

def load_mnist():
    with open('mnist.pkl', 'rb') as f:
        mnist = pickle.load(f)
    return mnist['training_images'], mnist['training_labels'], mnist['test_images'], mnist['test_labels']

train_x, train_y, test_x, test_y = load_mnist()

train_x, train_y, test_x, test_y = [pd.DataFrame(x) for x in [train_x, train_y, test_x, test_y]]

train_x = train_x/255.0
test_x = test_x/255.0

svc = SVC()

svc.fit(train_x, train_y.values.flatten())

filename = "svm_model.pkl"
pickle.dump(svc, open(filename, 'wb'))

y_pred = svc.predict(test_x)
print(classification_report(test_y, y_pred))