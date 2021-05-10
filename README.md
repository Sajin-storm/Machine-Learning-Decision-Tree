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

####The code is completely written using python language


## Code

### Necessary imports

```Python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
```




Here's a documentation for [plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)


