import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys
import math as mt
import random


random.seed(1)


# The node class
class Node:
   def __init__(self):
      self.parent = None
      self.left_child = None
      self.right_child = None
      self.tag = None
      self.best_feat = None

# This function returns the feature that maximize the mutual information gain
def return_best_feat(df, dict_cols,numb_of_boot, ncols):
   best_feat = ''
   best_h = sys.maxsize
   feature = []
   for i in range(numb_of_boot):
      x = random.randint(0,ncols-1)
      feature.append(x)

   for co in feature:
      colum = dict_cols[str(co)]
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
def train_recursive(data, node, currentDepth, maxDepth,dict_cols,numb_of_boot,ncols):
   if  (indicator_pure_node(data)):
      node.tag = return_best_prob(data)
#      print(indicator_pure_node(data))
      return 
   elif (currentDepth==maxDepth):
      node.tag = return_best_prob(data)
      return
   else:
      best_feature, cond_ent = return_best_feat(data,dict_cols,numb_of_boot,ncols)
      #print(cond_ent)
      inform = info_gain(data,cond_ent)
      if (currentDepth <= 2):
         print('The information gain for the ' + 'node ' + str(currentDepth) + ' is ' + str(inform))
         print('The best feature for the node ' + str(currentDepth) + ' is ' + str(best_feature))
      uniq = data[best_feature].unique()
#      print(uniq)
      uniq.sort()
#      print(uniq)
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
      train_recursive(data_1, node.left_child, currentDepth, maxDepth,dict_cols,numb_of_boot,ncols)
      train_recursive(data_2, node.right_child, currentDepth, maxDepth,dict_cols,numb_of_boot,ncols)
       


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
  # print(prob)
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
 #  test_node = node
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

# to measure the accuracy of the random forest classifier
def all_valid_nodes(data,node,numb_of_tree):
   nrows,ncols = data.shape
   pred = None
   y_true = data['Response']
   correct = 0.0
   my_dict = dict()
   for k in range(nrows):
      my_dict.clear()
      for i in range(numb_of_tree):
         predict = validate_tree(data,node[i],int(k))
	 #print(predict)
         if str(predict) in my_dict:
            my_dict[str(predict)]+=1
         else:
            my_dict[str(predict)]=1
      pred_value = -1
      value_at_key = -1
      for key in my_dict:
         if my_dict[key] > value_at_key:
            pred_value = int(key)
            value_at_key = my_dict[key]
      #print('the pred-value is ' + str(pred_value))
      if (y_true[k] == pred_value):
         correct+=1
   results = correct/nrows
   return results
      
      


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
df_valid = pd.concat([data_X_dev, data_Y_dev], axis=1)
rows,col = data_X.shape
dict_cols = dict()
for (i,k) in enumerate(df.columns):
   dict_cols[str(i)] = k
   
      

node = Node()
currentDepth = 0
maxDepth = 2
m = [5, 25, 50, 100]
T = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
accuracy = np.zeros([len(m),len(T)])

for (j,kk) in enumerate(m):
   numb_of_boot = kk
   array_node = []
   numb_of_trees = 100
   for i in range(numb_of_trees):
      node = Node()
      df_rand = df.sample(n = rows, replace = True) 
      train_recursive(df_rand, node, currentDepth, maxDepth, dict_cols,numb_of_boot,col)
      array_node.append(node)

   for (jj,k) in enumerate(T):
      nodes = np.random.choice(array_node, k, replace=False)
      acc = all_valid_nodes(df,nodes,k)
      accuracy[j,jj] = acc
      print(acc)



fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
plt.plot(T,accuracy[0,:],'o', linewidth=3, label = 'm=5')
plt.plot(T,accuracy[1,:],'o', linewidth=3, label = 'm=25')
plt.plot(T,accuracy[2,:],'o', linewidth=3, label = 'm=50')
plt.plot(T,accuracy[3,:],'o', linewidth=3, label = 'm=100')
plt.xlabel('T',fontsize = 16,fontweight='bold')
plt.ylabel('Accuracy',fontsize = 16,fontweight='bold')
plt.legend()
plt.title('Training accuracies')
plt.savefig('acc_dmax_2_training.png', format='png', dpi=500)
plt.show()


