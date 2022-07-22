import multivariate_cwru
import numpy
import pandas as pd

data = multivariate_cwru.CWRU("48DriveEndFault", 3600, 1, 1,1,'1750',normal_condition = True)

x_train = numpy.asarray(data.x_train).squeeze()
y_train = numpy.asarray(data.y_train)

df = pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train, columns=['labels'])], axis=1)

arr = [9, 11, 13]
df = df[df.labels.isin(arr) == True]

print(df.head())

df.to_csv("data.csv")

