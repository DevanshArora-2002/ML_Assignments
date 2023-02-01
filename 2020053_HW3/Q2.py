import pandas as pd
import numpy as np
import math
import gzip
filename="weights/part_B.sav"
import matplotlib.pyplot as plt
import pickle
def images_file_read(file_name):
  with gzip.open(file_name, 'r') as f:
    magic_number = int.from_bytes(f.read(4), 'big')
    image_count = int.from_bytes(f.read(4), 'big')
    row_count = int.from_bytes(f.read(4), 'big')
    column_count = int.from_bytes(f.read(4), 'big')
    image_data = f.read()
    images = np.frombuffer(image_data, dtype=np.uint8) \
      .reshape((image_count, row_count, column_count))
    return images
def labels_file_read(file_name):
  with gzip.open(file_name, 'r') as f:
    magic_number = int.from_bytes(f.read(4), 'big')
    label_count = int.from_bytes(f.read(4), 'big')
    label_data = f.read()
    labels = np.frombuffer(label_data, dtype=np.uint8)
    return labels
train_x = images_file_read("mnist/train-images-idx3-ubyte.gz")
print(train_x.shape)
train_y = labels_file_read("mnist/train-labels-idx1-ubyte.gz")
test_x = images_file_read("mnist/t10k-images-idx3-ubyte.gz")
print(test_x.shape)
test_y = labels_file_read("mnist/t10k-labels-idx1-ubyte.gz")


def flatten(train_x):
  lst = []
  for i in train_x:
    samp = i.reshape(i.shape[0] * i.shape[1])
    lst.append(samp)
  data = np.array(lst)
  return data
train_x = flatten(train_x)
test_x = flatten(test_x)
train_x=train_x.T
test_x=test_x.T
train_y=np.reshape(train_y,(1,train_y.shape[0]))
test_y=np.reshape(test_y,(1,test_y.shape[0]))
def format_y(data_y):
  tr_y=np.zeros(shape=(10,data_y.shape[1]))
  for i in range(data_y.shape[1]):
    label=data_y[0,i]
    tr_y[label,i]=1
  return tr_y
train_y_trans=format_y(train_y)
def save_model(list_models,filename):
  pickle.dump(list_models, open(filename, 'wb'))
def load_model(filename):
  list_models=pickle.load(open(filename, 'rb'))
  return list_models
def sigmoid(input):
  arr=np.exp(-1*input)
  ones=np.ones(shape=input.shape)
  arr=ones+arr
  arr=np.divide(ones,arr)
  return arr
def tanh(input):
  return np.tanh(input)
def relu(input):
  arr=np.where(input<0,0,input)
  return arr
def l_relu(input):
  arr=np.where(input<0,input*0.01,input)
  return arr
def softmax(input):
  base_sum=np.max(input,axis=0,keepdims=False)
  base_sum=np.reshape(base_sum,newshape=(1,input.shape[1]))
  inp=input-base_sum
  arr=np.exp(inp)
  sumval=np.sum(arr,axis=0)
  arr=np.divide(arr,sumval)
  return arr
def sigmoid_derv(input):
  inp=sigmoid(input)
  ones=np.ones(shape=input.shape)
  arr=np.multiply(inp,ones-inp)
  return arr
def tanh_derv(input):
  arr=np.exp(input)
  arr1=np.exp(-input)
  arr=arr+arr1
  arr=np.multiply(arr,arr)
  four=np.ones(shape=input.shape)
  four=4*four
  arr=np.divide(four,arr)
  return arr
def relu_derv(input):
  arr=input.copy()
  arr=np.where(arr<0,0,1)
  return arr
def l_relu_derv(input):
  arr=np.where(input<0,0.01,1)
  return arr
def softmax_derv(input):
  arr=softmax(input)
  arr=arr-np.multiply(arr,arr)
  return arr
def linear(input):
  return input
def linear_derv(input):
  return np.ones(input.shape)
def loss(proba,train_y):
  val=np.zeros(shape=proba.shape)
  for i in range(train_y.shape[1]):
    val[i,train_y[0,i]]=1
  val=val.T
  prob2=proba.T
  log1=np.log(prob2)
  pr=-1*np.multiply(val,log1)
  for i in range(pr.shape[0]):
    for j in range(pr.shape[1]):
      if(math.isnan(pr[i,j]) or math.isinf(pr[i,j])):
        pr[i,j]=0
  return np.sum(pr)/(val.shape[1])
def cross_loss_back_prop(val,target):
  fin=np.multiply(target,softmax(val)-1)
  sum=np.sum(fin,axis=0,keepdims=False)
  sum=np.reshape(sum,newshape=(1,target.shape[1]))
  return sum
def initialize(input,output,init,activ):
  data={}
  data['X']=np.zeros(shape=(input,output))
  data['Z']=np.zeros(shape=(output,1))
  data['A']=np.zeros(shape=(output,1))
  if(init=='zero_init'):
    data['weight']=np.zeros(shape=(output,input))
    data['bias']=np.zeros(shape=(output,1))
  elif(init=='random_init'):
    data['weight']=np.random.rand(output,input)
    data['weight']*=2
    data['weight']=data['weight']-1
    data['bias'] = np.zeros(shape=(output, 1))
  elif(init=='normal_init'):
    data['weight']=np.random.normal(0,1,size=(output,input))
    data['bias'] = np.zeros(shape=(output, 1))
  if(activ=='sigmoid'):
    data['activ']=sigmoid
    data['backprop']=sigmoid_derv
  elif(activ=='relu'):
    data['activ']=relu
    data['backprop']=relu_derv
  elif(activ=='l_relu'):
    data['activ']=l_relu
    data['backprop']=l_relu_derv
  elif(activ=='tanh'):
    data['activ']=tanh
    data['backprop']=tanh_derv
  elif(activ=='linear'):
    data['activ']=linear
    data['backprop']=linear_derv
  elif(activ=='softmax'):
    data['activ']=softmax
    data['backprop']=softmax_derv
  return data
class neural_network:
  def __init__(self,N,A,lr=0.001,activ=linear,init='zero_init',epoch=50,batch=100):
    self.lr=lr
    self.layers_data=[]
    self.layers_data.append(initialize(784,A[0],init,activ))
    for i in range(1,N):
      self.layers_data.append(initialize(A[i-1],A[i],init,activ))
    self.final_layer=initialize(A[-1],10,init,'softmax')
    self.epoch=epoch
    self.batch_size=batch
  def forward(self,input,lay):
    lay['X']=input
    lay['Z']=np.dot(lay['weight'],lay['X'])+lay['bias']
    lay['A']=lay['activ'](lay['Z'])
    return lay['A']
  def backward(self,loss,lay):
    multip=np.multiply(loss,lay['backprop'](lay['Z']))
    dw=np.dot(multip,lay['X'].T)
    db=np.sum(multip,axis=1,keepdims=True)
    ret=np.dot(lay['weight'].T,multip)
    return dw,db,ret
  def fit(self,input,target):
    no_samples=input.shape[1]
    for i in range(self.epoch):
      print(i)
      st=0
      while(st+self.batch_size<no_samples):
        inp=input[:,st:min(st+self.batch_size,no_samples)]
        for j in range(len(self.layers_data)):
          lay=self.layers_data[j]
          out=self.forward(inp,lay)
          inp=out
        out=self.forward(inp,self.final_layer)
        tar=target[:,st:min(st+self.batch_size,no_samples)]
        diff=out-tar
        dw=np.dot(diff,self.final_layer['X'].T)
        db=np.sum(diff,axis=1,keepdims=True)
        self.final_layer['weight']=self.final_layer['weight']-self.lr*(1/(no_samples-self.batch_size))*dw
        self.final_layer['bias']=self.final_layer['bias']-self.lr*(1/(no_samples-self.batch_size))*db
        loss=np.dot(self.final_layer['weight'].T,diff)
        for j in range(len(self.layers_data)-1,-1,-1):
          lay = self.layers_data[j]
          dw, db, loss = self.backward(loss, lay)
          lay['weight'] = lay['weight'] - self.lr * (1 / self.batch_size) * dw
          lay['bias'] = lay['bias'] - self.lr * (1 / self.batch_size) * db
        st+=self.batch_size
  def predict_proba(self,train_x):
    inp=train_x
    for lay in self.layers_data:
      out=self.forward(inp,lay)
      inp=out
    return self.forward(inp,self.final_layer)
  def predict(self,data_x):
    pred=[]
    for i in range(data_x.shape[1]):
      inp=data_x[:,i]
      inp=np.reshape(inp,(len(inp),1))
      for j in range(len(self.layers_data)):
        lay = self.layers_data[j]
        out = self.forward(inp, lay)
        inp = out
      out = self.forward(inp, self.final_layer)
      max_ind=0
      for j in range(10):
        if out[max_ind,0]<out[j,0]:
          max_ind=j
      pred.append(max_ind)
    return np.array([pred])
  def score(self,train_x,train_y):
    pred=self.predict(train_x)
    tot=train_y.shape[1]
    corr=0
    for i in range(tot):
      if pred[0,i]==train_y[0,i]:
        corr+=1
    return corr/tot
  def loss(self,proba, train_y):
    val = np.zeros(shape=proba.shape)
    for i in range(len(train_y)):
      val[i, train_y[i]] = 1
    val = val.T
    val2=1-val
    prob2 = proba.T
    log1 = np.log(prob2)
    log2 = np.log(1-prob2)
    pr = -1 * np.multiply(val, log1)
    pr2=-1*np.multiply(val2,log2)
    sum=np.sum(pr)+np.sum(pr2)
    return sum/(val.shape[1])
  def partial_fit(self,input,target):
    no_samples = input.shape[1]
    st = 0
    while (st + self.batch_size < no_samples):
      inp = input[:, st:min(st + self.batch_size, no_samples)]
      for j in range(len(self.layers_data)):
        lay = self.layers_data[j]
        out = self.forward(inp, lay)
        inp = out
      out = self.forward(inp, self.final_layer)
      tar = target[:, st:min(st + self.batch_size, no_samples)]
      diff = out - tar
      dw = np.dot(diff, self.final_layer['X'].T)
      db = np.sum(diff, axis=1, keepdims=True)
      self.final_layer['weight'] = self.final_layer['weight'] - self.lr * (1 / (no_samples - self.batch_size)) * dw
      self.final_layer['bias'] = self.final_layer['bias'] - self.lr * (1 / (no_samples - self.batch_size)) * db
      loss = np.dot(self.final_layer['weight'].T, diff)
      for j in range(len(self.layers_data) - 1, -1, -1):
        lay = self.layers_data[j]
        dw, db, loss = self.backward(loss, lay)
        lay['weight'] = lay['weight'] - self.lr * (1 / self.batch_size) * dw
        lay['bias'] = lay['bias'] - self.lr * (1 / self.batch_size) * db
      st += self.batch_size

train_x=train_x/255
test_x=test_x/255
params=[
    'sigmoid',
    'relu',
    'l_relu',
    'tanh',
    'linear'
]
models_info={}
for i in params:
    l1=[]
    l2=[]
    n_net=neural_network(N=4,A=[256,128,64,32],lr=0.01,activ=i,init='random_init',epoch=100,batch=128)
    for j in range(100):
        n_net.partial_fit(train_x,train_y_trans)
        prob1=n_net.predict_proba(train_x)
        prob2=n_net.predict_proba(test_x)
        l1.append(loss(prob1.T,train_y))
        l2.append(loss(prob2.T,test_y))
    x=[i for i in range(100)]
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(x,l1)
    plt.plot(x,l2)
    string="Plots_B/"+i+" loss"
    plt.savefig(string)
    plt.cla()
    models_info[i]=n_net
    print(i,end=" ")
    print("Training Loss {}".format(n_net.score(train_x,train_y)))
    print("Testing Loss {}".format(n_net.score(test_x,test_y)))
#n_net.fit(train_x,train_y_trans)
l1=[]
l2=[]
for i in range(5):
    n_net.partial_fit(train_x,train_y_trans)
    prob=n_net.predict_proba(train_x)
    prob2=n_net.predict_proba(test_x)
    l1.append(loss(prob.T,train_y))
    l2.append(loss(prob2.T,test_y))
x=[i for i in range(5)]
plt.plot(x,l1)
plt.plot(x,l2)
plt.show()
print(n_net.predict(train_x))
print(n_net.score(train_x, train_y))
print(n_net.score(test_x, test_y))
save_model(models_info,filename)