import pandas as pd
import numpy as np
x = pd.Series([1,2,3,4,5,6,7])
y = pd.Series([1,2,3,4,5,6,7])

#This is good if we have 2 variables to chrck
print(x.corr(y))


