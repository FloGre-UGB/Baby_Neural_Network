import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%%
df = pd.read_csv("college.csv")
y = df[["Private"]]
X = df.iloc[:,2:]

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.2, random_state = 1)

# initialize a neural network
L = 4 # no layers
# Define number of neurons per layer
n = [len(X.columns),4, 4, len(y["Private"].unique())]
# choose random weights and biases
np.random.seed(0)
Weights = dict()
biases = dict()
for i in range(0,len(n)-1):
    W = np.random.normal(0,1,n[i+1]*n[i])
    W = np.reshape(W, (n[i+1], n[i]))
    Weights[i+2] = W

    b = np.random.normal(0,1,n[i+1])
    b = np.transpose(b)
    biases[i+2] = np.reshape(b, (len(b),1))


delta = dict()
D = dict()
a = dict()
# Forward and Back Propagate
eta = 0.05 # learning rate
Niter = 10**4
cost = np.zeros(Niter)
for i in range(0,Niter):
    # choose a training point randomly
    k = np.random.randint(0, y_train.shape[0]-1)
    # feed it to the network
    x = np.array(X_train.iloc[k])
    a[1] = np.reshape(x, (len(x),1))
    # forward pass
    for l in range (2,L+1):
        prod = np.dot(Weights[l], a[l-1])
        prod = np.reshape(prod, (len(prod),1))
        z = prod + biases[l]
        z = np.reshape(z, (len(z),1))
        a[l] = sigmoid(z)
        a[l] = np.reshape(a[l], (len(a[l]),1)) # necessary to avoid dimension issues
        D[l] = np.diag(dsigmoid(z).flatten())
    # evaluate to which category observation k belongs
    y= np.reshape(np.array([y_train.iloc[k]=="Yes", y_train.iloc[k]=="No"]), (2,1))
    # this defines that neuron 0 in layer 4 represents "Yes" and neuron 1 represents "No"

    # backward pass
    delta[L] = np.dot(D[L],(a[L]-y))
    for l in range(L-1, 1, -1):
        delta[l] = np.dot(np.dot(D[l], np.transpose(Weights[l+1])), delta[l+1])

    # adjust weights and biases (gradient step)
    for l in range(L, 1, -1):
        #print(delta[l], a[l-1], np.outer(delta[l], a[l-1]))
        #print(l, a[l-1].shape, Weights[l].shape, np.outer(delta[l], a[l-1]).shape, delta[l].shape)
        Weights[l] = Weights[l] - eta * np.outer(delta[l], a[l-1])
        biases[l] = biases[l] - eta * delta[l]

    # calculate current Cost

    cost[i] = calculate_cost(y_train, X_train, Weights, biases, L)
#%%
# draw how the value of the cost function develops
fig, ax = plt.subplots()
ax.plot(np.arange(len(cost)),cost, label="Cost")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
fig.savefig("cost.png")


#%% make predictions
prediction = np.zeros(y_test.shape[0])
for i in range(0, y_test.shape[0]):
     prediction[i] = predict(forward_pass(X_test.iloc[i,:].values, Weights, biases, L))
y_test_logical = np.array([y_test["Private"] == "No"]).flatten()
#%%
cm = confusion_matrix(y_test_logical, y_test_logical)
cm_display = ConfusionMatrixDisplay(cm).plot()


#%%
def sigmoid(z):
    return 1/(1+np.exp(-z))
sigmoid = np.vectorize(sigmoid)

def dsigmoid(z):
    return sigmoid(z)* (1-sigmoid(z))
dsigmoid = np.vectorize(dsigmoid)

def calculate_cost(y_train, X_train, Weights, biases, L):
    Cost = 0
    # iterate over all training observations
    for i in range(0, y_train.shape[0]):
        a = X_train.iloc[i,:].values.reshape(-1,1)
        # feed the observation the NN
        for l in range(2,L+1):
            z = np.dot(Weights[l], a) + biases[l]
            a = sigmoid(z)
        # evaluate to which category observation i belongs
        y = np.reshape(np.array([y_train.iloc[i]=="Yes", y_train.iloc[i]=="No"]), (2,1))
        diff = y - a
        Cost = Cost + (np.linalg.norm(diff))**2
    return Cost

def forward_pass(X_i, Weights, biases, L):
    a = X_i.reshape(-1,1)
    for l in range(2,L+1):
        z = np.dot(Weights[l], a) + biases[l]
        a = sigmoid(z)
    return a

def predict(a):
    a = a.flatten()
    max_value = max(a)
    max_index = np.where(a == max_value)

    return max_index[0]

#%%
