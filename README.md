# Machine-Learning-Decision-Tree

>Disclaimer: I am just a student who has tried to study this on my own learning from various websites, documentation and vedios.
>Whatever is explained here is what I have understood myself. So, it may or maynot be correct .
>I'll provide with few links for some clarifications , if any furthur doubts arises you'll have to do your own research to clarify your own doubts.

## Table of contents

* [General info](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#general-info)
* [Technologies used](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#technologies-used)
* [Code](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#code)
	* [Necessary imports](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#necessary-imports)
	* [Reading Excel files](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#reading-excel-files)
	* [Seperating LMN values from the entire sheet](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#seperating-lmn-values-from-the-entire-sheet)
	* [Finding max values from each LM column](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#finding-max-values-from-each-lm-column)
	* [Creating a proper Dataframe for the test data](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#creating-a-proper-dataframe-for-the-test-data)
* [Machine learning parts](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#machine-learning-parts)
	* [Seperating features and classes for training and testing data](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#seperating-features-and-classes-for-training-and-testing-data)
	* [Decision Tree (ML algorithm)](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#seperating-features-and-classes-for-training-and-testing-data)
	* [Plots and Diagrams](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#plots-and-diagrams)
	* [Getting the final result](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#getting-the-final-result)
	* [Finally arranging the values in descending order](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree#getting-the-final-result)

## General info

#### we have a table with some data 
* we need to seperately get only the LMN values columns 
* Find the max in each column
* Then we need to classify them into 3 groups
	* non-critical
	* semi-critical
	* critical
*Finally we need to sort it in descending order

I have used Decision Tree Classifier to classify 
because it provides visual support and 
it's very easy to understand.

## Technologies used
Project is created with :
* Spyder (Python 3.8)

Test and Train Data is present in :
* Excel (.xlsx format)

#### The code is completely written using python language


## Code

### Necessary imports

```Python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
```
### Reading Excel files

Pandas provides support to read data from excel sheets
```Python
dataset_train = pd.read_excel('TrainData.xlsx')
```
We can also use the below code to read different sheets within the given excel sheet

```Python
df1 = pd.read_excel('TestData.xlsx', sheet_name=['Line Outage (n-1)'])
```

You can visit the documumentation page for [pd.read_excel](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html) if you want to know more 

### Seperating LMN values from the entire sheet

If you can see from TestData.xlsx it doesn't contain only values related to what we need.
It also has other values which we do not need. So, we need to extract the values which we need.

```Python
df1 = pd.read_excel('TestData.xlsx', sheet_name=['Line Outage (n-1)']) #reading entire data
#The above line is of type Dictionary
df = df1['Line Outage (n-1)'].head(41)  #cutting it down by 41 rows because the data we need is present only in 41 rows
#The above line is of type DataFrame
table = {}    #empty dictionary. 

for i in range(1,42):        #for loop to iterate 41 times                
    table['Lm'+str(i)] = df['Lm'+str(i)]  # find columns which begins with Lm and add it to the dictionary

dataset_test_lm = pd.DataFrame(table)   #convert the dictionary to dataframe
```
You can go through these link's if you aren't aware of terms 
[Dictionary](https://www.w3schools.com/python/python_dictionaries.asp) 
and [DataFrame](https://www.w3schools.com/datascience/ds_python_dataframe.asp).

### Finding max values from each LM column 

This is very simple. Just 1 line of code to find the max of each column.

```Python
col_max = dataset_test_lm.max()
```

Sample Output

```
#Sample output for col_max
Lm1	0.101473
Lm2 	0.055696
Lm3 	0.087793
Lm4 	0.030696
Lm5 	0.182670
```

### Creating a proper Dataframe for the test data

Now that we got the max values of each LM column. We can now make it better looking by creating a DataFrame with Sl.no, LM column number and it's values

```Python
Sl_no = []	#empty list
for x in range(1,len(col_max)+1):   #iterating throuhgh length of col_max to get Sl.no
    Sl_no.append(x)

lm_head = []	#empty list
for x in col_max.index:		#iterating through col.max index values to get LM column numbers
    lm_head.append(int(x[2:]))	#Using String slicing removed LM and kept only number and also converted string to int
    
lm_values = []	#empty list
for x in col_max:		#iterating through col_max itself would provide its values
    lm_values.append(x)
```

Now we have list of Sl.no , LM and values. we can merge them all together into a dictonary and convert it into a DataFrame

```Python
data = {'Sl.no':Sl_no , 'LM':lm_head , 'values': lm_values}
dataset_test = pd.DataFrame(data, columns = ['Sl.no','LM','values'])
```

Now we get our output something like this

```
       Sl.no    LM    	 values
0       1   	1	0.101473
1       2   	2	0.055696
2       3   	3	0.087793
3       4   	4	0.030696
4       5   	5	0.182670
5       6   	6	0.136450
```

## Machine learning parts

### Seperating features and classes for training and testing data

Extracting features and class for training data. Here I have only values as the only features, if there is many features we can add as many as we need.
And also the data is converted into array using numpy.

```Python
#we need to add features which would decide the possible output (class)
features_train = dataset_train[['values']]  #we only have values as our feature
#Sl.no is not a feature to determine our output
#class is our output so no need to add it to features

x_train = np.asarray(features_train)	#converting featues to array
y_train = np.asarray(dataset_train['class'])	#getting classes and converting them to array
#classes are non-critical, semi-critical and critical
```
similar way for testing data as well

```Python
features_test = dataset_test[['values']]

x_test = np.asarray(features_test)

#here we only have features because we need to find class using ML
```

### Decision Tree (ML algorithm)

sklearn provides this machine learning algorithm. we import DecisionTreeClassifier from sklearn.tree.
* And to be honest I don't know much of what is happening here

```Python
#initialsing Decision Tree Classifier
classifier = DecisionTreeClassifier()
#Providing Training data to Classifier
classifier.fit(x_train,y_train)
#Trying to predict the Output by Providing testing data
y_predict = classifier.predict(x_test)

```

### Plots and Diagrams

I've tried to get Plots for training data and testing data. Also the tree diagram for visualisation of whats happening.
* matplotlib.pyplot provides features for plotting graphs.
* sklearn.tree.plot_tree provides feature for plotting the tree diagram or flowchart.

Here is the code for plotting Training data. For Test data look into my [code](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree/blob/main/DecisionTree_ML.py)

```Python
### distribution of classes ###
semi_critical_train = dataset_train[dataset_train['class'] == 'semi-critical']	#getting all semi-critical values
critical_train = dataset_train[dataset_train['class'] == 'critical']		#getting all critical values
non_critical_train = dataset_train[dataset_train['class'] == 'non-critical']	#getting all non-critical values

###plotting train graphs###
axes = semi_critical_train.plot(kind='scatter', x='Sl.no', y='values', color='blue', label='semi-critical' ,title = 'Train Data Plot')
critical_train.plot(kind='scatter', x='Sl.no', y='values', color='red', label='critical' ,ax=axes)
non_critical_train.plot(kind='scatter', x='Sl.no', y='values', color='green', label='non-critical', ax=axes)

```

#### We get this kind of output plot. 
![Test Plot](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree/blob/main/Sample%20output%20images/Train_Plot.png?raw=true)

Now for the tree diagram

```Python

plt.figure(figsize=(30,20))
a = plot_tree(classifier, 
              feature_names = features_test.columns,	#the feature column heading names
              class_names = result['class'].unique().tolist(), 	#classes 
              filled=True, rounded=True, fontsize=30, node_ids=True)
```

#### We get this kind of tree diagram. 
![Tree_Diagram](https://github.com/Sajin-storm/Machine-Learning-Decision-Tree/blob/main/Sample%20output%20images/Tree_Diagram.png?raw=true)

Here's the documentation for [plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html) it'll be very helpful

### Getting the final result

So now the DecisionTreeCLassifier would have done it's job of classifying by it's own. We can see by what margin the classification has happened by seeing the tree diagram.

y_predict will have the final output. So,I created a DataFrame with the output

```Python
sl_no_test = []		#empty array
for i in range(1,len(y_predict_list)+1):	#similar iteration to get range of values for Sl.no
    sl_no_test.append(i)

data_list = {'Sl.no': sl_no_test ,'LM':lm_head, 'values': lm_values , 'class': y_predict_list}	#creating dictionary with Sl.no, LM, values and class that we got
result = pd.DataFrame(data_list , columns = ['Sl.no', 'LM', 'values', 'class'])		#converting dictionary to DataFrame

```

Now we have our output which we got from the code by itself. If we want we can save it as a new excel sheet, I just kept it as it is.

Here's the output sample

```

	Sl.no  LM    	   values        class
0       1	1	  0.101473      critical
1       2   	2	  0.055696  	semi-critical
2       3   	3	  0.087793      critical
3       4   	4	  0.030696   	non-critical
4       5   	5	  0.182670      critical
5       6   	6	  0.136450      critical

```

### Finally arranging the values in descending order

I didn't know how to arrange a DataFrame in order. So created a new Dataframe and added values in descending order

```Python
result_order = result.sort_values(by=['values'], ascending=False)
result_order_data = {'Sl.no':sl_no_test, 'LM':result_order['LM'].tolist(), 'values':result_order['values'].tolist(), 'class':result_order['class'].tolist()}
result_order_dataframe = pd.DataFrame(result_order_data, columns = ['Sl.no', 'LM', 'values', 'class'])
print(result_order_dataframe)
```

and here's the sample output

```
    Sl.no  LM    values          class
0       1  12  0.323621       critical
1       2  20  0.228563       critical
2       3  22  0.213246       critical
3       4  15  0.196902       critical
4       5   5  0.182670       critical
5       6  17  0.179817       critical
.
.
.
35     36  14  0.022431   non-critical
36     37  13  0.020233   non-critical
37     38  27  0.018241   non-critical
38     39  26  0.012403   non-critical
39     40  41  0.011116   non-critical
40     41  29  0.006553   non-critical

```



