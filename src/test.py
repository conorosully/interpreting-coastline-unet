import numpy as np
import pandas as pd 

x = [1,2,3,4,5,6,7,8,9,10]
y = [2,4,6,8,10,12,14,16,18,20]


dict = {"x":x,"y":y}

df = pd.DataFrame(dict)

df.to_csv("test.csv")
