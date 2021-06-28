import pandas as pd
import requests
from collections import Counter
import numpy as np
import pprint
eps = np.finfo(float).eps

#Importing necessary packages

#Class which helps us read our dataset from csv file and replace '?' with most common value for that atrribute

class DataReader:
  def __init__(self, url):
    self.feature_set = []
    self.data = []
    self.list_of_most_common_value = []
    self.set_of_lines = []

    for i in range(22):
      feature_value_counter = Counter()
      self.feature_set.append(feature_value_counter)

    r = requests.get( url, stream=True )

    for line in r.iter_lines():
      line = line.decode('utf-8')
      self.set_of_lines.append(line)

  def find_most_common_values(self):
    
    for line in self.set_of_lines:
      line_values = line.split(',')
      
      if len(line_values) != 23: 
        continue
      
      y_label = line_values[0]
      x_atributes = line_values[1:]
      
      for i in range(22):
        attribute_val = x_atributes[i]
        if attribute_val !='?':
          self.feature_set[i][attribute_val]+=1

    for f in self.feature_set:
      most_frequent_value = f.most_common()[0][0]
      self.list_of_most_common_value.append(most_frequent_value)
    
  def read(self):

    self.find_most_common_values()
    
    for line in self.set_of_lines:
      
      line_values = line.split(',')
      
      if len(line_values) != 23: 
        continue

      y_label = [line_values[0]]
      x_atributes = line_values[1:]
      
      for i in range(22):
        attribute_val = x_atributes[i]
        most_common_attribute_value = self.list_of_most_common_value[i]
        if attribute_val =='?':
          x_atributes[i]=most_common_attribute_value
      
      updated_data = y_label + x_atributes
      self.data.append(updated_data)

train_data_url = 'https://raw.githubusercontent.com/jeniyat/CSE-5521-SP21/master/HW/HW1/Data/train.csv'
dr = DataReader(train_data_url)
dr.read()

#Using Panda to convert it into dataframe
train_data_frame = pd.DataFrame(dr.data)

#Function to calculate overall entropy of the system for a particular dataframe

def entropy(train_data_frame):
    entropy=0
    
    #The decision_column which is the first column in our dataframe that contains information about edibility of a mushroom
    decision_column=train_data_frame.keys()[0]

    #Attribute Values for the decision column which are 'e' and 'p'
    attribute_values = train_data_frame[decision_column].unique()

    #Calculating entropy using given formula
    for values in attribute_values:
        entropy=entropy-(train_data_frame[decision_column].value_counts()[values]/len(train_data_frame[decision_column]))*np.log2(train_data_frame[decision_column].value_counts()[values]/len(train_data_frame[decision_column]))
    return entropy

#Calculate entropy of attributes in the give dataframe

def entropy_attribute(data_frame,attribute):

    #The decision_column which is the first column in our dataframe that contains information about edibility of a mushroom
    decision_column= data_frame.keys()[0]

    #Attribute Values for the decision column which are 'e' and 'p'
    dc_values=data_frame[decision_column].unique()  

    #Attribute Values for the given attribute column
    values = data_frame[attribute].unique()

    total_entropy = 0

    #Calculating total entropy of the given attribute column 
    for value in values:
        entropy = 0

        #Calculating entropy for all the attribute values of the given attribute column
        for dc_value in dc_values:
                total_attribute_values=len(data_frame[attribute][data_frame[attribute]==value])
                particular_attribute_value=len(data_frame[attribute][data_frame[attribute]==value][data_frame[decision_column]==dc_value])

                probability_attribute=particular_attribute_value/(total_attribute_values+eps)
                entropy = entropy-probability_attribute*np.log2(probability_attribute+eps)

        probability_system=total_attribute_values/len(data_frame)
        total_entropy =total_entropy-probability_system*entropy
    return abs(total_entropy)

#Getting a subset of a root_node

def subset(data_frame,node,value):

    #Getting the dataframe of a particular attribute value of a particular node/attribute for a particular decision column value
    subset=data_frame[value==data_frame[node]].reset_index(drop=True)
    return subset

#Creating our ID3 decision tree

def ID3_tree(data_frame): 

  #The decision_column which is the first column in our dataframe that contains information about edibility of a mushroom
  decision_column = data_frame.keys()[0] 

  #List which stores info gains of attributes and their attribute values
  info_gain_values=[]

  #Calculating info gain for a particular attribute
  for attributes in data_frame.keys()[1:]:
      info_gain_values.append(entropy(data_frame)-entropy_attribute(data_frame,attributes))

  #Getting the index/attribute which has the max info gain with respect to its parent node and the decision column
  node=data_frame.keys()[1:][np.argmax(info_gain_values)]

  #Attribute Values for a particular attribute column
  attribute_values = np.unique(data_frame[node])  

  #Creating an ID3 node                
  id3_tree={}
  id3_tree[node] = {}

  for value in attribute_values:

      root_subset=subset(data_frame,node,value)
      counts=entropy(root_subset)

      #Checking if the node is homogeneous i.e. its entropy is 0
      if counts==0:
          id3_tree[node][value] = np.unique(root_subset[0])[0]                                                 
      else:        
          id3_tree[node][value] = ID3_tree(root_subset)
                    
  return id3_tree

#Creating the ID3 decision tree from our training set data

id3_tree = ID3_tree(train_data_frame)

#Creating the prediction function that will predict the edibility of a mushroom from a particular test_data row

def prediction(test_data_row,id3_tree):

    #Getting the nodes of our tree
    for nodes in id3_tree.keys(): 
        prediction_value = 0     

        #Getting the attribute value of the attribute node present in our tree from the test_data row
        value = test_data_row[nodes]
        id3_tree = id3_tree[nodes][value]
        
        #Checking if the tree needs to be further traversed
        if type(id3_tree) is dict:
            prediction_value = prediction(test_data_row, id3_tree)
        else:
            prediction_value = id3_tree
            break;                            
        
    return prediction_value

#Printing our ID3 Decision Tree

pprint.pprint(id3_tree)

#Getting our test_data from test_data csv file

test_data_url = 'https://raw.githubusercontent.com/jeniyat/CSE-5521-SP21/master/HW/HW1/Data/test.csv'
test_data = DataReader(test_data_url)
test_data.read()

test_df = pd.DataFrame(test_data.data)

#Decision Column of our test_dataset
testY_label=test_df[0]

#Getting the attributes of the mushroom in our test_dataset
testX_label=test_df.drop(0,axis=1)

counter=0

#Calculating accuracy score of our decision tree
for i in range(1623):
  num=prediction(testX_label.iloc[i],id3_tree)
  if(num==testY_label[i]):
    counter=counter+1
accuracy=counter/(len(testY_label)-1)

print('%.2f'%accuracy)