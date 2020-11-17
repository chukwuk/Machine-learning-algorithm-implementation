import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys

data = pd.read_csv('PA1_train.csv')
## part a
df = data.drop(columns = ['id'])
## end of part a
data_validate = pd.read_csv('PA1_dev.csv')
nrows,ncols =  df.shape
df_validate = data_validate.drop(columns = ['id'])

m = np.zeros([nrows,1])
p = df['price'].values
k = df['date'].values
d = []
m = []
yy = []

k_valid = df_validate['date'].values
d_valid = []
m_valid = []
yy_valid = []
#part b for splitting the feature
for i in range(len(k)):
   x,y,z = k[i].split('/')
   d.append(int(x))
   m.append(int(y))
   yy.append(int(z))

for i in range(len(k_valid)):
   aa, bb, cc = k_valid[i].split('/')
   d_valid.append(int(aa))
   m_valid.append(int(bb))
   yy_valid.append(int(cc))


df =  df.drop(columns = 'date')
df.insert(1,'month',d)
df.insert(2,'day',m)
df.insert(3,'year',yy)

df_validate =  df_validate.drop(columns = 'date')
df_validate.insert(1,'month',d_valid)
df_validate.insert(2,'day',m_valid)
df_validate.insert(3,'year',yy_valid)

# end of part b


# part e
#cols_price = df['price']
#max_value = max(cols_price)
#min_value = min(cols_price)

#new_value = []
#for x in cols_price:
#   vals = (x - min_value)/(max_value - min_value)
#   new_value.append(round(vals,2))
#part e
cate_feat = ['dummy','waterfront','grade','condition','view','year','price']
for colum in df.columns:
   if colum not in cate_feat:
      cols_zipcode = df[colum]
      max_value_z = max(cols_zipcode)
      min_value_z = min(cols_zipcode)
      new_value_z =[]
      for y in cols_zipcode:
         valsz = (float(y) - float(min_value_z))/(float(max_value_z) - float(min_value_z))
         new_value_z.append(round(valsz,2))

      df[colum] = new_value_z

for colum in df_validate.columns:
   if colum not in cate_feat:
      cols_zipcode = df_validate[colum]  
      max_value_z = max(cols_zipcode)
      min_value_z = min(cols_zipcode)
      new_value_z =[]
      for y in cols_zipcode:
         valsz = (float(y) - float(min_value_z))/(float(max_value_z) - float(min_value_z))
         new_value_z.append(round(valsz,2))

      df_validate[colum] = new_value_z


## to map 2014 to 0  and 2015 to 1 in the year column
df['year'].replace(to_replace=[int(2014), int(2015)], value=[0, 1], inplace = True)
df_validate['year'].replace(to_replace=[int(2014), int(2015)], value=[0, 1], inplace = True)

## dropping the month and day for the purchase of the house 
df = df.drop(columns = ['day', 'month'])
df_validate =  df_validate.drop(columns = ['day', 'month'])
#print(df_validate.shape)
#COPYING THE features vectors and outcomes in X and Y respectively
Y1 = df['price'].values
Y = np.zeros([nrows,1])
Y_V = df_validate['price'].values
nrows_v, ncols_v = df_validate.shape
Y_V_1 = np.zeros([nrows_v,1])



for i in range(nrows):
   Y[i] = float(Y1[i])



for k in range(nrows_v):
   Y_V_1[k] = float(Y_V[k])

X_train = df.iloc[:,0:20].values
X_train  = np.asarray(X_train)
Y_train = np.asarray(Y)
Y_test = np.asarray(Y_V_1)
X_test = df_validate.iloc[:,0:20].values

nrows_v, ncols_v = X_test.shape


nrows, ncols = X_train.shape

for i in range(nrows):
   for k in range(ncols):
      X_train[i,k] = float(X_train[i,k]) 

for i in range(nrows_v):
   for k in range(ncols_v):
      X_test[i,k] = float(X_test[i,k])


nrows,ncols = X_train.shape

#nrows_v, ncols_v = X.test.shape

W = np.zeros([ncols,1])
for i in range(ncols):
   if i !=0:
      #W[i] = 1.0
      W[i] = 0.02
   else:
      W[i]  = 0.2
      #W[i]  = 0.0


#l_rate = 0.00000000005
l_rate = 0.001
i = 0
n = 5
temp = 0.0
uu = 0
MSE = []
num_of_iter = []

#The batch gradient descent algorithm
while n > 0:
   KK = np.matmul(X_train,W)
   sub = np.subtract(KK,Y_train)
   fin = np.matmul((np.transpose(X_train)),sub)
   fin_1 = (float(2)/float(nrows))*fin 
   squ_err = np.power(sub,2)
   sum_squ_err = np.sum(squ_err )
   err_mse = (sum_squ_err)/float(nrows)
   MSE.append(err_mse)
   num_of_iter.append(i)
   W1 = l_rate*fin_1
   W = np.subtract(W,W1)
   i+=1
   if LA.norm(fin_1) < 0.5:
      break
   if LA.norm(fin_1) >=sys.maxsize:
      print('it is diverging at iteration ' + str(i))
      break


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



