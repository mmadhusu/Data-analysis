#!/usr/bin/python
import sys
import csv
import pandas as pd
import numpy as np
from pandas import DataFrame

#f = open('train-manip.csv')
#3csv_f = csv.reader(f)
#ow0 = next(csv_f)
#row0.append('hour')

df = pd.read_csv('test-grid-placeid.csv', index_col = 'row_id')
#print df.head()
#df ['Hour'] = (df ['time']/100)%24
#df ['Weekday'] = (df ['time']/(60*24))%7
#df ['Month'] = (df ['time']/(60*24*30))%12
#df ['Year'] = df ['time']/(60*24*365)
#df ['Day'] = df ['time']/(60*24) % 365
#print df.head()

#grid = df [(df['x'] > 1.7) & (df ['x'] < 2.0) & (df['y'] > 1.2) & (df['y'] < 1.3)]
#print grid

#Code to add a new column with no values at the end of the dataframe
#df ['place_id'] = ' '
#print df.head()
#df.to_csv ('Train-full-feature.csv')
# This works
#Adding a new column at a particular index value
df.insert(4, 'place_id',' ')
#print df.head()
#df.to_csv ('test-grid-placeid.csv')

#try to get unique values in the column
#g = df.groupby('place_id').place_id.nunique()
#print g

#Drop a column from a dataframe by using the column
#del df['place_id']
print df.head()
df.to_csv('test-grid-placeid.csv')

 


