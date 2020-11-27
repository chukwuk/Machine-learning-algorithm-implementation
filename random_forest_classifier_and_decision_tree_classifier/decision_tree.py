import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys
import math as mt
import random

# The node class
class Node:
   def __init__(self):
      self.parent = None
      self.left_child = None
      self.right_child = None
      self.tag = None
      self.best_feat = None

# This function returns the feature that maximize the mutual information gain
def return_best_feat(df):
   best_feat = ''
   best_h = sys.maxsize
   for colum in df.columns:
      if colum == 'Response':
         continue
      uniq = df[colum].unique()
      summ = len(df[colum])
      hx = 0.0
      for key in uniq:
         tt = df[df[colum]==int(key)]
         tot = len(tt['Response'])
         pb = float(tot)/float(summ)
         my_dict_1 = dict()
         my_dict_1.clear()
         for k in tt['Response']:
            if str(k) in my_dict_1:
               my_dict_1[str(k)]+= 1
            else:
               my_dict_1[str(k)] = 1
         summa = 0.0
         for ke in my_dict_1:
            summa+=my_dict_1[ke]
         proba = 0.0
         for kk in my_dict_1:
            prob = float(my_dict_1[kk])/float(summa)
            proba+=(-1* prob * mt.log(prob,2))
         proba*=pb
         hx+=proba
      if hx < best_h:
         best_h = hx
         best_feat = colum 
   
   return (best_feat,best_h)  
  
# trains the random forest classifier
def train_recursive(data, node, currentDepth, maxDepth):
   if  (indicator_pure_node(data)):
      node.tag = return_best_prob(data)
      return 
   elif (currentDepth==maxDepth):
      node.tag = return_best_prob(data)
      return
   else:
      best_feature, cond_ent = return_best_feat(data)
      inform = info_gain(data,cond_ent)
      if (currentDepth <= 2):
         print('The information gain for the ' + 'node ' + str(currentDepth) + ' is ' + str(inform))
         print('The best feature for the node ' + str(currentDepth) + ' is ' + str(best_feature))
      uniq = data[best_feature].unique()
      uniq.sort()
      if (len(uniq)==1):
         node.tag = return_best_prob(data)
         return
      currentDepth+=1      
      node.best_feat = best_feature
      node.left_child = Node()
      node.right_child = Node()
      node.left_child.parent = node;
      node.right_child.parent = node;
      data_1 = data[data[best_feature]==int(uniq[0])]
      data_2 = data[data[best_feature]==int(uniq[1])]
      train_recursive(data_1, node.left_child, currentDepth, maxDepth)
      train_recursive(data_2, node.right_child, currentDepth, maxDepth)
       


# return the class label with the highest probability
def return_best_prob (df):
   my_dict_1 = dict()
   my_dict_1.clear()
   best_tag = -1
   tag_num = -1
   for k in df['Response']:
      if str(k) in my_dict_1:
         my_dict_1[str(k)]+= 1
      else:
         my_dict_1[str(k)] = 1
   for key in my_dict_1:
      if (best_tag <  my_dict_1[key]):
         tag_num = int(key)
         best_tag = my_dict_1[key]
      if (best_tag == my_dict_1[key]):
         x = random.randint(0,1)
         if x==0:
            best_tag = best_tag
            tag_num = tag_num
         else:
            tag_num = int(key)
            best_tag = my_dict_1[key]

   return tag_num  


# returns the information gain
def info_gain(data,best_feat):
   my_dict_1 = dict()
   prob = 0.0
   for i in data['Response']:
     if str(i) in my_dict_1:
        my_dict_1[str(i)]+=1
     else:
        my_dict_1[str(i)] = 1
   summa = 0.0
   for key in my_dict_1:
      summa+=my_dict_1[key]
   for k in my_dict_1:
      p = float(my_dict_1[k])/float(summa)
      prob+=(-1* p * mt.log(p,2))
   return (prob-best_feat)    
     
# function to check if a node is pure
def indicator_pure_node(df):
   uniq = df['Response'].unique()
   if (len(uniq)  == 1):
      return True
   else:
      return False

# function  to test one data point  using one tree
def validate_tree(data,node,x):
   tag =None
   if (node.tag == None):
      feature = node.best_feat
      s = data[feature]
      value = s[x]
      if (value == 0):
        tag = validate_tree(data, node.left_child,x)
      else:
        tag =  validate_tree(data, node.right_child,x)
      return tag
   else:
      return node.tag

# function to test the accuracy of a validation data
def all_valid_tree(data,node):
   nrows,ncols = data.shape
   pred = None
   y_true = data['Response']
   correct = 0.0
   for i in range(nrows):
      pred = validate_tree(data,node,int(i))
      if (y_true[i] == pred):
         correct+=1
   result = correct/nrows
   return result

      
      


data_X = pd.read_csv('pa4_train_X.csv')
## part a
data_Y =  pd.read_csv('pa4_train_y.csv')

data_X_dev = pd.read_csv('pa4_dev_X.csv')
## part a
data_Y_dev =  pd.read_csv('pa4_dev_y.csv')


## end of part a

nrows,ncols =  data_X.shape
nrows_v, ncols_v = data_X_dev.shape


#print(data_X_dev.iloc[:3])



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


nrows,ncols = X_train.shape



W = np.zeros([ncols,1])
W_av = np.zeros([ncols,1])
for i in range(ncols):
   W[i] = 0.0
   W_av[i] = 0.0



# initialize list of lists 
data = [[1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0]] 
  
# Create the pandas DataFrame 
#df = pd.DataFrame(data, columns = ['test_1', 'test_2', 'Response']) 

df = pd.concat([data_X, data_Y], axis=1)
#print(df) 
df_valid = pd.concat([data_X_dev, data_Y_dev], axis=1)
#print(random.randint(0,1))
#print(random.randint(0,1))
#print(return_best_feat(df))
node = Node()
currentDepth = 0
maxDepth = 20
#print(info_gain(df,0.5))
#train_recursive(df, node, currentDepth, maxDepth)
#print(node.best_feat)
#print(node.right_child.best_feat)
#print(all_valid_tree(df,node))
#print(all_valid_tree(df_valid,node))
dmax = [2,5,10,20,25,30,35,40,45,50]
accuracy = []
accuracy_valid = []
#print(dmax)

for i in dmax:
   node = Node()
   currentDepth = 0
   maxDepth = i
   train_recursive(df, node, currentDepth, maxDepth)
   accu = all_valid_tree(df,node)
   accu_valid = all_valid_tree(df_valid,node)
   accuracy.append(accu)
   accuracy_valid.append(accu_valid)



fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
plt.plot(dmax,accuracy,'s', linewidth=3, label = 'Training set')
plt.plot(dmax,accuracy_valid,'o', linewidth=3, label = 'Validation set')
plt.xlabel('dmax',fontsize = 16,fontweight='bold')
plt.ylabel('Accuracy',fontsize = 16,fontweight='bold')
plt.savefig('acc_dmax_1.png', format='png', dpi=500)
plt.legend()
plt.show()


