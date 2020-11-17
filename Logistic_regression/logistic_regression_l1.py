import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys


def sigmoid(z,u):
   nrows, ncols = z.shape
   result = np.zeros([nrows, ncols])
   for x in range(nrows):
      if u == 0:
         result[x,ncols-1] = ( 1/(1+np.exp(-z[x,ncols-1]) ))
      else:
         result[x,ncols-1] = (np.exp(-z[x,ncols-1])/(1.+np.exp(-z[x,ncols-1]) ))
   return result


def accuracy_data_pred(Y_train, pred_val):
   nrows, ncols = Y_train.shape
   acc = 0;
   for y in range(nrows):
      if Y_train[y,ncols-1]==np.around(pred_val[y,ncols-1]):
         acc+=1
   result = acc/nrows
   return result
         

data_X = pd.read_csv('pa2_train_X.csv')
## part a
data_Y =  pd.read_csv('pa2_train_y.csv')

data_X_dev = pd.read_csv('pa2_dev_X.csv')
## part a
data_Y_dev =  pd.read_csv('pa2_dev_y.csv')


## end of part a

nrows,ncols =  data_X.shape
nrows_v, ncols_v = data_X_dev.shape


cate_feat = ['Age','Annual_Premium','Vintage']
for colum in data_X.columns:
   if colum  in cate_feat:
      cols_zipcode = data_X[colum]
      max_value_z = max(cols_zipcode)
      min_value_z = min(cols_zipcode)
      new_value_z =[]
      for y in cols_zipcode:
         valsz = (float(y) - float(min_value_z))/(float(max_value_z) - float(min_value_z))
         new_value_z.append(round(valsz,2))

      data_X[colum] = new_value_z

for colum in data_X_dev.columns:
   if colum  in cate_feat:
      cols_zipcode = data_X_dev[colum]  
      max_value_z = max(cols_zipcode)
      min_value_z = min(cols_zipcode)
      new_value_z =[]
      for y in cols_zipcode:
         valsz = (float(y) - float(min_value_z))/(float(max_value_z) - float(min_value_z))
         new_value_z.append(round(valsz,2))

      data_X_dev[colum] = new_value_z




#COPYING THE features vectors and outcomes in X and Y respectively



Y_train = np.zeros([nrows,1])
Y_test = np.zeros([nrows_v,1])

Y = data_Y['Response']
Y_1 = data_Y_dev['Response']

for i in range(nrows):
   Y_train[i] = float(Y[i])


X_train = np.asarray(data_X)

for i in range(nrows_v):
   Y_test[i] = float(Y_1[i])

X_test = np.asarray(data_X_dev)

#print(Y_train)

#for i in range(nrows):
#   for k in range(ncols):
#      X_train[i,k] = float(X_train[i,k]) 

#for i in range(nrows_v):
#   for k in range(ncols_v):
#      X_test[i,k] = float(X_test[i,k])


nrows,ncols = X_train.shape


#print(data_Y.shape) 

W = np.zeros([ncols,1])
for i in range(ncols):
   if i !=0:
      #W[i] = 0.00000000001
      W[i] = 0.0
   else:
      W[i] = 0.0
     # W[i]  = 0.0000000000001

#U = [2,2,2]
U = np.zeros([3,1])
U[0] =0.5
U[1] = 0.88
U[2] = 0.2
P = np.zeros([3,1])
P[0:] = 0.0
#print(W.shape)
result = sigmoid(U,0)
#result = np.asarray(result)
#print(accuracy_data_pred(P,U))
#print(result.shape)
#print(sigmoid(U,0))


l_rate = 0.01
reg_par = 0.00001
i = 0
n = 5
temp = 0.0
uu = 0
MSE = []
num_of_iter = []

#The gradient descent for ridge logistic regression algorithm
while n > 0:
   KK = np.matmul(X_train,W)
   KK_1 = sigmoid(KK,0)
   #print(KK_1[1])
   sub = np.subtract(Y_train,KK_1)
   fin = np.matmul((np.transpose(X_train)),sub)
   fin_1 = (l_rate/float(nrows))*fin

   num_of_iter.append(i)
   W = np.add(W,fin_1)
   #print(W)
   for k in range(len(W)-1):
      W[k]= np.abs(W[k])-(l_rate*reg_par*W[k])
      if (W[k] < 0.0):
         W[k] = 0.0

  # print(W)
   WtX = np.matmul(X_train,W)
   pred_val = sigmoid(WtX,0)
   acc = accuracy_data_pred(Y_train,pred_val)
   print (acc)
 #  print(i)
   i+=1
   if i >= 1500:
      break
   if LA.norm(fin_1) >=sys.maxsize:
      print('it is diverging at iteration ' + str(i))
      break

KK_test = np.matmul(X_test,W)
KK_test_1 = sigmoid(KK_test,0)
acc_test = accuracy_data_pred(Y_test,KK_test_1)
print('The accuracy of the validation data:' + str(acc_test))


sparsity_count = 0


for k in range(len(W)):
   if W[k] == 0.0:
     sparsity_count+=1

percent_sparsity = (float(sparsity_count)/float(ncols))*100.0
print('The number of zero count is:' + str(sparsity_count))
print('The percentage sparsity is:' + str(percent_sparsity))

print(W[4])
print(W[196])
print(W[86])
print(W[60])
print(W[35])


#W = [-0.00003,0.000001,0.000002,0.000004,0.0000006]
W = np.transpose(W)


uu = open('Weight_1.txt', 'w')
for kk in range(len(W)):
 # if kk!=5 and kk!=7:
    uu.write("%s" %W[kk])


W = np.abs(W)
kk = np.argsort(W)
print(kk)

#print(np.sort(W))
#print(u)
# The best features
print(data_X.iloc[:,[4,196,86,60,35]])



"""
loss_1 = np.matmul(X_test,W)
loss_2 = np.subtract(loss_1,Y_test)
squ_err_valid = np.power(loss_2,2)
sum_squ_err_valid = np.sum(squ_err_valid)
err_mse_valid = sum_squ_err_valid/float(nrows_v)
print('The MSE for the validation data is: ' + str(err_mse_valid))
print('The MSE for the training data is: ' + str(MSE[i-1]))

#print(df.columns)

#fig = plt.figure(figsize=(7,7))
#ax = fig.add_subplot(111)
#plt.plot(num_of_iter,MSE,'b-', linewidth=3)
#plt.xlabel('number of iteration',fontsize = 16,fontweight='bold')
#plt.ylabel('MSE',fontsize = 16,fontweight='bold')
#plt.savefig('lr_0_00001.png', format='png', dpi=500)
#plt.show()



print('The number of iteration is: ' + str(i))
print('Below is the co-efficient')
print(W)
   

#print(kk)
#p = np.zeros(2)
#p[1] = 1
#print(W[2])
#print(p[~0])


"""
