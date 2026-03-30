import numpy
import keyboard
from sklearn.svm import SVC
import joblib
x = numpy.load("data.npy")
y = numpy.load("labels.npy")
print(len(x)) 
clf = SVC() 
clf.fit(x, y)
clf.fit(x, y)
correct = 0.0
for i in range(0, len(x) - 1):      
    prid = clf.predict(x)[i]
    if prid == y[i]:
        correct += 1.0
        
joblib.dump(clf,"clf.pkl")
print(correct / len(x))
    
        