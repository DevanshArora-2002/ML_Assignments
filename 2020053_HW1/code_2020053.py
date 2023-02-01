import pandas as pd
import numpy as np
import openpyxl
from sklearn.manifold._t_sne import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition._pca import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn import preprocessing
import sklearn.metrics as met
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
def l1_slope(w):
    row=w[:,0]
    row=row.T
    lst=[]
    for i in row:
        if i>0:
            lst.append(1)
        else:
            lst.append(-1)
    arr=np.array(lst)
    arr.reshape((len(arr),1))
    return arr
def train_test_split(data_x,data_y,test_size):
    no_train_samples=int(data_x.shape[1]*(1-test_size))
    train_x=data_x[:,0:no_train_samples]
    train_y=data_y[0:no_train_samples]
    test_x=data_x[:,no_train_samples:]
    test_y=data_y[no_train_samples:]
    return train_x,train_y,test_x,test_y
def optimize(data_x):
    for i in range(data_x.shape[0]):
        maxi=np.amax(data_x[i,:])
        for j in range(data_x.shape[1]):
            data_x[i,j]=float(data_x[i,j])/float(maxi)
def mae(pred_y,data_y):
    loss=0
    for i in range(len(data_y)):
        loss+=abs(data_y[i]-pred_y[0,i])
    return loss/len(data_y)
def mse(pred_y,data_y):
    loss=0
    for i in range(len(data_y)):
        loss+=(data_y[i]-pred_y[0,i])**2
    return loss/len(data_y)
def rmse(pred_y,data_y):
    rmse_loss=mse(pred_y,data_y)**0.5
    return rmse_loss
class LinearRegression():
    def __init__(self,x):
        self.num_features=x.shape[0]
        self.w=np.ones((self.num_features,1))
        self.mse=[]
        self.rmse=[]
        self.mae=[]
    def set_init(self):
        self.w=np.ones((self.num_features,1))
    def normalize(self,x):
        mean=np.mean(x,axis=1)
        mean=np.reshape(mean,[self.num_features,1])
        return x-mean
    def fit(self,x,y,epoch,learning_rate,reg=None):
        for i in range(epoch):
            y_hat=np.dot(self.w.T,x)
            diff=np.dot(x,(y_hat-y).T)
            diff=diff/(x.shape[1])
            if(reg=='l2'):
                self.w=self.w-learning_rate*(diff+self.w/x.shape[1])
            if(reg=='l1'):
                self.w=self.w-learning_rate*(diff+l1_slope(self.w)/x.shape[1])
            else:
                self.w=self.w-learning_rate*diff
    def plot(self,x,y,epoch,learning_rate):
        y_lis=[]
        for i in range(epoch):
            y_hat=np.dot(self.w.T,x)
            diff=np.dot(x,(y_hat-y).T)
            diff=diff/(x.shape[1])
            self.w=self.w-learning_rate*diff
            y_lis.append(i)
            self.f_rmse(x,y)
        return self.rmse,y_lis
    def predict(self,x):
        pred_y=np.dot(self.w.T,x)
        #np.reshape(pred_y,(len(pred_y)))
        return pred_y
    def f_mae(self,data_x,data_y):
        pred_y=np.dot(self.w.T,data_x)
        loss=0
        for i in range(len(data_y)):
            loss+=abs(data_y[i]-pred_y[0,i])
        self.mae.append(loss/len(data_y))
    def f_mse(self,data_x,data_y):
        pred_y=np.dot(self.w.T,data_x)
        loss=0
        for i in range(len(data_y)):
            loss+=(data_y[i]-pred_y[0,i])**2
        self.mse.append(loss/len(data_y))
    def f_rmse(self,data_x,data_y):
        pred_y=np.dot(self.w.T,data_x)
        loss=0
        for i in range(len(data_y)):
            loss+=(data_y[i]-pred_y[0,i])**2
        self.rmse.append((loss/len(data_y))**0.5)
def k_fold_norm(data_x,data_y,folds,lin_reg):
    lst = []
    lst_y = []
    no_div=data_x.shape[1]
    invid_fold_size=no_div//folds
    left_over=no_div%folds
    lst.append(data_x[:,0:left_over+invid_fold_size])
    lst_y.append(data_y[0:left_over+invid_fold_size])
    i=left_over+invid_fold_size
    while(i<=no_div-invid_fold_size):
        lst.append(data_x[:,i:i+invid_fold_size])
        lst_y.append(data_y[i:i+invid_fold_size])
        i=i+invid_fold_size
    for i in range(len(lst)):
        train_x=np.empty((data_x.shape[0],0))
        train_y=np.empty((0))
        test_x=lst[i]
        test_y=lst_y[i]
        for j in range(len(lst)):
            if(j!=i):
                train_x=np.concatenate((train_x,lst[i]),axis=1)
                train_y=np.concatenate((train_y,lst_y[i]))
        lin_reg.set_init()
        lin_reg.fit(train_x,train_y,epoch=1000,learning_rate=0.1)
        lin_reg.f_mse(test_x,test_y)
        lin_reg.f_mae(test_x,test_y)
        lin_reg.f_rmse(test_x,test_y)
def plot_k(data_x,data_y,folds,epochs,reg=None):
    lst=[]
    lst_y=[]
    no_div = data_x.shape[1]
    invid_fold_size = no_div // folds
    left_over = no_div % folds
    lst.append(data_x[:, 0:left_over + invid_fold_size])
    lst_y.append(data_y[0:left_over + invid_fold_size])
    i = left_over + invid_fold_size
    while (i <= no_div - invid_fold_size):
        lst.append(data_x[:, i:i + invid_fold_size])
        lst_y.append(data_y[i:i + invid_fold_size])
        i = i + invid_fold_size
    for i in range(len(lst)):
        train_x=np.empty((data_x.shape[0],0))
        train_y=np.empty((0))
        test_x=lst[i]
        test_y=lst_y[i]
        for j in range(len(lst)):
            if(j!=i):
                train_x=np.concatenate((train_x,lst[i]),axis=1)
                train_y=np.concatenate((train_y,lst_y[i]))
        lin_mod=LinearRegression(data_x)
        lst_loss1 = []
        lst_loss2 = []
        for e in range(epochs):
            lin_mod.fit(train_x,train_y,1,0.01,reg)
            pred_t_y=lin_mod.predict(train_x)
            pred_val_y=lin_mod.predict(test_x)
            lst_loss1.append(rmse(pred_t_y,train_y))
            lst_loss2.append(rmse(pred_val_y,test_y))
        y=range(epochs)
        plt.plot(y,lst_loss1)
        plt.title("Training Loss {}".format(i))
        plt.show()
        plt.plot(y,lst_loss2)
        plt.title("Training Loss {}".format(i))
        plt.show()
def plot_reg(train_x,train_y,test_x,test_y,epochs,reg=None):
    lin_reg=LinearRegression(train_x)
    lst=[]
    train_err=[]
    test_err=[]
    for i in range(epochs):
        lst.append(i)
        lin_reg.fit(train_x,train_y,1,0.1,reg)
        train_err.append(rmse(lin_reg.predict(train_x),train_y))
        test_err.append(rmse(lin_reg.predict(test_x),test_y))
    plt.plot(lst,train_err)
    plt.plot(lst,test_err)
    plt.title("{} Loss ".format(reg))
    plt.legend(['Train-error','Test-error'])
    plt.show()
def normal_impl(data_x,data_y):
    weight=np.linalg.inv(np.dot(data_x,data_x.T))
    we2=np.dot(data_x,data_y)
    weight=np.dot(weight,we2)
    return weight
def k_fold_(data_x,data_y,folds):
    lst = []
    lst_y = []
    lst_rmse=[]
    no_div = data_x.shape[1]
    invid_fold_size = no_div // folds
    left_over = no_div % folds
    lst.append(data_x[:, 0:left_over + invid_fold_size])
    lst_y.append(data_y[0:left_over + invid_fold_size])
    i = left_over + invid_fold_size
    while (i <= no_div - invid_fold_size):
        lst.append(data_x[:, i:i + invid_fold_size])
        lst_y.append(data_y[i:i + invid_fold_size])
        i = i + invid_fold_size
    for i in range(len(lst)):
        train_x=np.empty((data_x.shape[0],0))
        train_y = np.empty((0))
        test_x = lst[i]
        test_y = lst_y[i]
        for j in range(len(lst)):
            if(j!=i):
                train_x=np.concatenate((train_x,lst[i]),axis=1)
                train_y=np.concatenate((train_y,lst_y[i]))
        weight=normal_impl(train_x,train_y)
        pred_y=np.dot(weight.T,test_x)
        pred_y=pred_y.reshape((pred_y.shape[0],1))
        pred_y=pred_y.T
        lst_rmse.append(rmse(pred_y,test_y))
    return lst_rmse
print("Section B")
df=pd.read_csv('Real_estate.csv')
df.columns=['No','X1','X2','X3','X4','X5','X6','Y']
x_columns=df.columns[1:-1]
data_x=df[x_columns]
data_x=data_x.to_numpy()
data_x=data_x.T
ones=np.ones(data_x.shape[1])
data_x=np.vstack((data_x,ones))
data_y=df[df.columns[-1]].to_numpy()
data_x_t=data_x[:]
optimize(data_x)
lst_rmse=[]
k_val=[]
for i in range(2,6):
    lin_mod=LinearRegression(data_x)
    k_fold_norm(data_x, data_y, i,lin_mod)
    lst_rmse.append(np.sum(np.array(lin_mod.rmse))/len(lin_mod.rmse))
    k_val.append(i)
print("Loss against each values of k")
print(lst_rmse)
print(k_val)
plot_k(data_x,data_y,5,epochs=1000)
train_x,train_y,test_x,test_y=train_test_split(data_x,data_y,test_size=0.1)
plot_reg(train_x,train_y,test_x,test_y,epochs=1000,reg='l1')
plot_reg(train_x,train_y,test_x,test_y,epochs=1000,reg='l2')
lst_rmse=k_fold_(data_x,data_y,5)
print("RMSE of respective val set")
print(lst_rmse)
print("Section C")
df=pd.read_excel('Dry_Bean_Dataset.xlsx',sheet_name=0)
sns.countplot(data=df,x=df.columns[-1])
plt.show()
for i in range(len(df.columns[0:-1])):
    plt.subplot(2,2,1)
    sns.boxplot(data=df,x=df.columns[i])
    plt.subplot(2, 2, 2)
    sns.barplot(data=df,x=df.columns[-1],y=df.columns[i])
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df,y=df.columns[i],x=df.columns[-1])
    plt.subplot(2, 2, 4)
    sns.pointplot(data=df,x=df.columns[-1],y=df.columns[i])
    plt.show()
sns.heatmap(data=df.corr(),annot=True)
plt.show()
print("-------------")
print("Columns with the number of null values")
print(df.isna().sum())

data=df.to_numpy()
data_x=data[:,0:-1]
data_y=data[:,-1]

data_tr=TSNE(n_components=2,perplexity=10,n_iter=250).fit_transform(data_x)
uni_class=np.unique(data_y)
dict={}
for i in uni_class:
    dict[i]=[[],[]]
    dict[i]=[[],[]]
for i in range(len(data_y)):
    dict[data_y[i]][0].append(data_x[i][0])
    dict[data_y[i]][1].append(data_x[i][1])
for i in dict:
    plt.scatter(dict[i][0],dict[i][1])
plt.legend([i for i in dict])
plt.show()
mult=MultinomialNB()
train_x,test_x,train_y,test_y=tts(data_x,data_y,test_size=0.2)
mult.fit(train_x,train_y)
pred_y=mult.predict(test_x)
print("Multinomial Naive bayes metrics")
print("The Precision is {}".format(met.precision_score(test_y,pred_y,average='macro')))
print("The Accuracy is {}".format(met.accuracy_score(test_y,pred_y)))
print("The Recall is {}".format(met.recall_score(pred_y,test_y,average='macro')))
guass=GaussianNB()
guass.fit(train_x,train_y)
pred_y=guass.predict(test_x)
print("------------")
print("Gaussian Naive bayes metrics")
print("The precision is {}".format(met.precision_score(test_y,pred_y,average='macro')))
print("The Accuracy is {}".format(met.accuracy_score(test_y,pred_y)))
print("The recall is {}".format(met.recall_score(pred_y,test_y,average='macro')))

model=LogisticRegression(multi_class='multinomial',max_iter=100)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
data_y=data_y.reshape(len(data_y),1)
pca=PCA(n_components=4)
print("No components 4")
data_tr=pca.fit_transform(data_x)
train_x,test_x,train_y,test_y=tts(data_tr,data_y,test_size=0.2)
model=LogisticRegression(multi_class='multinomial',max_iter=100)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
print("Accuracy is {}".format(met.accuracy_score(pred_y,test_y)))
print("Precision is {}".format(met.precision_score(pred_y,test_y,average='macro')))
print("Recall is {}".format(met.recall_score(pred_y,test_y,average='macro')))
print("F1 is {}".format(met.f1_score(pred_y,test_y,average='macro')))
print("---------------")
pca=PCA(n_components=6)
print("No of components 6")
data_tr=pca.fit_transform(data_x)
train_x,test_x,train_y,test_y=tts(data_tr,data_y,test_size=0.2)
model=LogisticRegression(multi_class='multinomial',max_iter=100)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
print("Accuracy is {}".format(met.accuracy_score(pred_y,test_y)))
print("Precision is {}".format(met.precision_score(pred_y,test_y,average='macro')))
print("Recall is {}".format(met.recall_score(pred_y,test_y,average='macro')))
print("F1 is {}".format(met.f1_score(pred_y,test_y,average='macro')))
print("---------------")
pca=PCA(n_components=8)
print("No of components 8")
data_tr=pca.fit_transform(data_x)
train_x,test_x,train_y,test_y=tts(data_tr,data_y,test_size=0.2)
model=LogisticRegression(multi_class='multinomial',max_iter=100)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
print("Accuracy is {}".format(met.accuracy_score(pred_y,test_y)))
print("Precision is {}".format(met.precision_score(pred_y,test_y,average='macro')))
print("Recall is {}".format(met.recall_score(pred_y,test_y,average='macro')))
print("F1 is {}".format(met.f1_score(pred_y,test_y,average='macro')))
print("---------------")
pca=PCA(n_components=10)
print("No of components 10")
data_tr=pca.fit_transform(data_x)
train_x,test_x,train_y,test_y=tts(data_tr,data_y,test_size=0.2)
model=LogisticRegression(multi_class='multinomial',max_iter=100)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
print("Accuracy is {}".format(met.accuracy_score(pred_y,test_y)))
print("Precision is {}".format(met.precision_score(pred_y,test_y,average='macro')))
print("Recall is {}".format(met.recall_score(pred_y,test_y,average='macro')))
print("F1 is {}".format(met.f1_score(pred_y,test_y,average='macro')))
print("---------------")
pca=PCA(n_components=12)
print("No of components 12")
data_tr=pca.fit_transform(data_x)
train_x,test_x,train_y,test_y=tts(data_tr,data_y,test_size=0.2)

model=LogisticRegression(multi_class='multinomial',max_iter=100)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
print("Accuracy is {}".format(met.accuracy_score(pred_y,test_y)))
print("Precision is {}".format(met.precision_score(pred_y,test_y,average='macro')))
print("Recall is {}".format(met.recall_score(pred_y,test_y,average='macro')))
print("F1 is {}".format(met.f1_score(pred_y,test_y,average='macro')))

train_x,test_x,train_y,test_y=tts(data_x,data_y,test_size=0.2)
model=LogisticRegression(multi_class='multinomial',max_iter=100)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
print("------------")
print("Logistic Regression metrics")
print("The precision is {}".format(met.precision_score(test_y,pred_y,average='macro')))
print("The Accuracy is {}".format(met.accuracy_score(test_y,pred_y)))
print("The recall is {}".format(met.recall_score(pred_y,test_y,average='macro')))
y_bin=label_binarize(test_y,classes=np.unique(test_y))
false_pos_r={}
true_pos_r={}
th={}
auc_val={}
pred_prob=model.predict_proba(test_x)
unique_cl=np.unique(test_y)
for i in range(len(unique_cl)):
    false_pos_r[i],true_pos_r[i],th[i]=roc_curve(y_bin[:,i],pred_prob[:,i])
    auc_val[i]=auc(false_pos_r[i],true_pos_r[i])
    plt.plot(false_pos_r[i],true_pos_r[i])
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.legend([i for i in unique_cl])
plt.show()

