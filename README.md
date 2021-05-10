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

Here's a documentation for [plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)


