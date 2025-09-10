import numpy as np
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

iris = load_iris()
x,y = iris.data, iris.target

np.random.seed(8675311)  # comment for random splits
i = np.argsort(np.random.random(len(y)))
x,y = x[i],y[i]

xtrn, ytrn = x[:100], y[:100]
xtst, ytst = x[100:], y[100:]

#np.random.seed()  # uncomment to randomly initialize
mlp = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=2000)
mlp.fit(xtrn,ytrn)
prob = mlp.predict_proba(xtst)
pred = np.argmax(prob, axis=1)

nc = 0
for i in range(len(ytst)):
    print("%0.4f %0.4f %0.4f  %d  %d  %s" % (prob[i,0],prob[i,1],prob[i,2],pred[i],ytst[i], pred[i]==ytst[i]))
    nc += 1 if (pred[i]==ytst[i]) else 0
print()
print("Accuracy = %0.6f" % (nc/len(ytst),))
print()

cm = np.zeros((3,3), dtype="uint8")
for i in range(len(ytst)):
    cm[ytst[i],pred[i]] += 1
print(cm)
print()

