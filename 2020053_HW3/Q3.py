import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import train_test_split
filename="weights/Part_C.sav"
from sklearn.metrics import log_loss
import pickle
def save_model(list_models,filename):
  pickle.dump(list_models, open(filename, 'wb'))
def load_model(filename):
  list_models=pickle.load(open(filename, 'rb'))
  return list_models
def images_file_read(file_name):
    with gzip.open(file_name, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images
def labels_file_read(file_name):
    with gzip.open(file_name, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels
train_x = images_file_read("fmnist/train-images-idx3-ubyte.gz")
print(train_x.shape)
train_y = labels_file_read("fmnist/train-labels-idx1-ubyte.gz")
test_x = images_file_read("fmnist/t10k-images-idx3-ubyte.gz")
print(test_x.shape)
test_y = labels_file_read("fmnist/t10k-labels-idx1-ubyte.gz")
def plot(train_loss,val_loss,name):
  x=[i+1 for i in range(len(train_loss))]
  plt.xlabel("No of epochs")
  plt.ylabel("Loss")
  plt.plot(x,train_loss)
  plt.plot(x,val_loss)
  string = "Plots/" + "loss"+str(name)
  plt.savefig(string)
  plt.cla()
def flatten(train_x):
  lst=[]
  for i in train_x:
    samp=i.reshape(i.shape[0]*i.shape[1])
    lst.append(samp)
  data=np.array(lst)
  return data
def loss(proba,train_y):
  val=np.zeros(shape=proba.shape)
  for i in range(len(train_y)):
    val[i,train_y[i]]=1
  val=val.T
  prob2=proba.T
  log1=np.log(prob2)
  pr=-1*np.multiply(val,log1)
  return np.sum(pr)/(val.shape[1])
train_x=flatten(train_x)
test_x=flatten(test_x)
mlp1=MLP(activation='logistic',hidden_layer_sizes=(256,32),max_iter=100,batch_size=300,learning_rate_init=0.001)
mlp2=MLP(activation='relu',hidden_layer_sizes=(256,32),max_iter=100,batch_size=300,learning_rate_init=0.001)
mlp3=MLP(activation='identity',hidden_layer_sizes=(256,32),max_iter=100,batch_size=300,learning_rate_init=0.001)
mlp4=MLP(activation='tanh',hidden_layer_sizes=(256,32),max_iter=100,batch_size=300,learning_rate_init=0.001)
def fit(mlp,train_x,train_y,test_x,test_y,name):
    train_loss = []
    val_loss = []
    mlp.partial_fit(train_x, train_y, classes=np.unique(train_y))
    prob_tr = mlp.predict_proba(train_x)
    prob_val = mlp.predict_proba(test_x)
    train_loss.append(mlp.loss_)
    val_loss.append(loss(prob_val, test_y))
    for i in range(50):
        mlp.partial_fit(train_x, train_y)
        prob_tr = mlp.predict_proba(train_x)
        prob_val = mlp.predict_proba(test_x)
        train_loss.append(mlp.loss_)
        val_loss.append(log_loss(y_true=test_y, y_pred=prob_val,labels=np.unique(test_y)))
    plot(train_loss, val_loss,name)
    print("Type {}".format(name),end=" ")
    print(mlp1.score(test_x, test_y))
fit(mlp1,train_x,train_y,test_x,test_y,'logistic')
fit(mlp1,train_x,train_y,test_x,test_y,'relu')
fit(mlp1,train_x,train_y,test_x,test_y,'identity')
fit(mlp1,train_x,train_y,test_x,test_y,'tanh')
list_models={}
list_models['activation']={
    'sigmoid':mlp1,
    'relu':mlp2,
    'linear':mlp3,
    'tanh':mlp4
}
mlp5=MLP(activation='relu',hidden_layer_sizes=(256,32),max_iter=100,batch_size=300,learning_rate_init=0.1)
mlp6=MLP(activation='relu',hidden_layer_sizes=(256,32),max_iter=100,batch_size=300,learning_rate_init=0.01)
mlp7=MLP(activation='relu',hidden_layer_sizes=(256,32),max_iter=100,batch_size=300,learning_rate_init=0.001)
fit(mlp5,train_x,train_y,test_x,test_y,'rate1')
fit(mlp6,train_x,train_y,test_x,test_y,'rate2')
fit(mlp7,train_x,train_y,test_x,test_y,'rate3')
list_models['learn_rate']={
    '0.1':mlp5,
    '0.01':mlp6,
    '0.001':mlp7
}
mlp8=MLP(activation='relu',hidden_layer_sizes=(128,32),max_iter=100,batch_size=300,learning_rate_init=0.01)
mlp9=MLP(activation='relu',hidden_layer_sizes=(64,32),max_iter=100,batch_size=300,learning_rate_init=0.01)
mlp10=MLP(activation='relu',hidden_layer_sizes=(32,16),max_iter=100,batch_size=300,learning_rate_init=0.01)
fit(mlp8,train_x,train_y,test_x,test_y,'layer1')
fit(mlp9,train_x,train_y,test_x,test_y,'layer2')
fit(mlp10,train_x,train_y,test_x,test_y,'layer3')
list_models['layer']={
    '128':mlp8,
    '64':mlp9,
    '32':mlp10
}
from sklearn.model_selection import GridSearchCV
mlp=MLP(batch_size=300,hidden_layer_sizes=(256,32),max_iter=50)
params = {
    'activation': ['relu','logistic','identity','tanh'],
    'learning_rate_init': [0.1,0.01,0.001]
}
grid_search = GridSearchCV(estimator=mlp,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")
grid_search.fit(train_x,train_y)
rf_best = grid_search.best_estimator_
print(rf_best)
fit(rf_best,train_x,train_y,test_x,test_y,"Best")
list_models['best']=rf_best
save_model(list_models,filename)
