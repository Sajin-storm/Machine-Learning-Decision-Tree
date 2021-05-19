# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:48:18 2021

@author: sajin
"""

###necessary imports###
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

### Training Dataset ###

dataset_train = pd.read_excel('TrainData.xlsx')

print('Train Data :\n')
print(dataset_train)


### Testing Dataset ###

### extracting only LM values ###
df1 = pd.read_excel('TestData.xlsx', sheet_name=['Line Outage (n-1)'])
df = df1['Line Outage (n-1)'].head(41)
table = {}

for i in range(1,42):
    table['Lm'+str(i)] = df['Lm'+str(i)]

dataset_test_lm = pd.DataFrame(table)

### finding max LM value from every LM column ###

col_max = dataset_test_lm.max(axis=0) #axis=0 is for finding max values column wise

### creating testing dataset ###

Sl_no = []
for x in range(1,len(col_max)+1):
    Sl_no.append(x)

lm_head = []
for x in col_max.index:
    lm_head.append(int(x[2:]))
    
lm_values = []
for x in col_max:
    lm_values.append(x)
    
data = {'Sl.no':Sl_no , 'Line number':lm_head , 'LMN values': lm_values}

dataset_test = pd.DataFrame(data, columns = ['Sl.no','Line number','LMN values'])

print('\nTest data:\n')
print(dataset_test)


### machine learning parts ###

### distribution of classes ###

semi_critical_train = dataset_train[dataset_train['class'] == 'semi-critical']
critical_train = dataset_train[dataset_train['class'] == 'critical']
non_critical_train = dataset_train[dataset_train['class'] == 'non-critical']

###plotting train graphs###
axes = semi_critical_train.plot(kind='scatter', x='Line number', y='LMN values', color='blue', title = 'Train Data Plot')
critical_train.plot(kind='scatter', x='Line number', y='LMN values', color='red', ax=axes)
non_critical_train.plot(kind='scatter', x='Line number', y='LMN values', color='green',  ax=axes)

semi_critical_train.plot(x='Line number', y='LMN values', color='blue', label='semi-critical' ,title = 'Train Data Plot', ax=axes)
critical_train.plot( x='Line number', y='LMN values', color='red', label='critical' ,ax=axes)
non_critical_train.plot( x='Line number', y='LMN values', color='green', label='non-critical', ax=axes)

### identifying and removing unwanted data ###

#print(dataset_train.columns) ### will return a list containing heading 

###train data
features_train = dataset_train[['LMN values']]

x_train = np.asarray(features_train)
y_train = np.asarray(dataset_train['class'])

###test data
features_test = dataset_test[['LMN values']]

x_test = np.asarray(features_test)

### Decision Tree ###

classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_predict = classifier.predict(x_test)


### results ###

y_predict_list = y_predict.tolist()

sl_no_test = []
for i in range(1,len(y_predict_list)+1):
    sl_no_test.append(i)

data_list = {'Sl.no': sl_no_test ,'Line number':lm_head, 'LMN values': lm_values , 'class': y_predict_list}
result = pd.DataFrame(data_list , columns = ['Sl.no', 'Line number', 'LMN values', 'class'])

print('\nResult:\nClassification by ML\n')
print(result)

print('\nIn Descending Order:\n')
result_order = result.sort_values(by=['LMN values'], ascending=False)
result_order_data = {'Sl.no':sl_no_test, 'Line number':result_order['Line number'].tolist(), 'LMN values':result_order['LMN values'].tolist(), 'class':result_order['class'].tolist()}
result_order_dataframe = pd.DataFrame(result_order_data, columns = ['Sl.no', 'Line number', 'LMN values', 'class'])
print(result_order_dataframe)



### distribution of classes ###
semi_critical_test = result[result['class'] == 'semi-critical']
critical_test = result[result['class'] == 'critical']
non_critical_test = result[result['class'] == 'non-critical']

###plotting test graphs###
axes = semi_critical_test.plot(kind='scatter', x='Line number', y='LMN values', color='blue', title = 'Test Data Plot')
critical_test.plot(kind='scatter', x='Line number', y='LMN values', color='red', ax=axes)
non_critical_test.plot(kind='scatter', x='Line number', y='LMN values', color='green', ax=axes)

"""
plt.axhline(y = 0.049, color = 'blue', linestyle = '-')
plt.axhline(y = 0.116, color = 'red', linestyle = '-')
# rendering the plot
plt.show()
"""

semi_critical_test.plot( x='Line number', y='LMN values', color='blue', label='semi-critical' ,title = 'Test Data Plot', ax=axes)
critical_test.plot( x='Line number', y='LMN values', color='red', label='critical' ,ax=axes)
non_critical_test.plot( x='Line number', y='LMN values', color='green', label='non-critical', ax=axes)

result.plot(kind="bar", x='Line number', y='LMN values')
plt.show()

"""
### bar graph###
semi_critical_test.plot.bar( x='Line number', y='LMN values', color='blue', label='semi-critical' ,title = 'Test Data Plot')
critical_test.plot(kind='bar', x='Line number', y='LMN values', color='red', label='critical' )
non_critical_test.plot.bar( x='Line number', y='LMN values', color='green', label='non-critical')
"""

###bar graph ###
plt.bar( semi_critical_test['Line number'], semi_critical_test['LMN values'], color='blue',label='semi-critical')
plt.bar( critical_test['Line number'], critical_test['LMN values'], color='red', label='critical')
plt.bar( non_critical_test['Line number'], non_critical_test['LMN values'], color='green', label='non-critical')
plt.legend()
plt.show()

### visualising decision tree ###

result_classifier = DecisionTreeClassifier()
result_classifier.fit(x_test,y_predict)

plt.figure(figsize=(30,20))
a = plot_tree(result_classifier, 
              feature_names = features_test.columns,
              class_names = ['critical','non-critical','semi-critical'],#result['class'].unique().tolist(), 
              filled=True, rounded=True, fontsize=30, node_ids=True, impurity=(False))
plt.show()


