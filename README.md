# Machine-Learning-Decision-Tree

## General info

#### we have a table with some data 
* we need to seperately get only the LM values columns 
* Find the max in each column
* Then we need to classify them into 3 groups
	* non-critical
	* semi-critical
	* critical
*Finally we need to sort it in descending order

I have used Decision Tree Classifier to classify 
because it provides visual support and 
it's very easy to understand.

## Technologies
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
We can also use it to read different sheets within the given excel sheet

```Python
df1 = pd.read_excel('TestData.xlsx', sheet_name=['Line Outage (n-1)'])
```

You can visit the documumentation page for [pd.read_excel](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html) if you want to know more 

### Seperating LM values from the entire sheet

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

Sl.no|LM|values
---|-----|---------
1.|Lm1|0.101473
2.|Lm2|0.055696
3.|Lm3|0.087793
4.|Lm4|0.030696
5.|Lm5|0.182670
     

Here's a documentation for [plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)


