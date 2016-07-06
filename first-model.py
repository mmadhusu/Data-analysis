#!/usr/bin/python
import sys
import csv
import pandas as pd
import numpy as np
from pandas import DataFrame

def data_prep (input_csv,x_cord1,x_cord2,y_cord1,y_cord2):
	
        #Read the csv file
	df = pd.read_csv(input_csv, index_col = 'row_id')
  
        #Create data grid
        grid = df [(df['x'] > x_cord1) & (df ['x'] < x_cord2) & (df['y'] > y_cord1) & (df['y'] < y_cord2)]

	#Feature engineering : Creating the features
	grid ['Hour'] = (grid ['time']/60)%24
	grid ['Weekday'] = (grid ['time']/(60*24))%7
	grid ['Month'] = (grid ['time']/(60*24*30))%12
	grid ['Year'] = grid ['time']/(60*24*365)
	grid ['Day'] = grid ['time']/(60*24) % 365
	grid.to_csv ('test-grid.csv')
	print grid

	#time provides the same info. Delete the time variable
        del grid['time']

	#Feature Engineering : Assigning the weights for each feature
        feature_weights =  [500, 1000, 4, 3, 1./22., 2, 10]
        grid ['x'] = grid.x * feature_weights[0]
        grid ['y'] = grid.y * feature_weights[1]
        grid['Hour'] = grid.Hour * feature_weights[2]
        grid['Weekday'] = grid.Weekday * feature_weights[3]
        grid['Day'] = (grid.Day * feature_weights[4]).astype(int)
        grid['Month'] = grid.Month * feature_weights[5]
        grid['Year'] = grid.Year * feature_weights[6]

	return grid


def train_prep (train_grid)	
	#Keep the cutoff of place id values as k=10
	#Any other place_id values are just dropped from the grid
	cut_off = 10
	placeid_count = train_grid.place_id.value_counts() #Counts the number of unique occurences of a place id
	flag = (placeid_count[train_grid.place_id.values] > 0).values # Adds boolean if the number of values > cutoff
	modified_train_grid = train_grid.loc[flag] #Considers only the place_ids that occur more than the cutoff times
	return modified_train_grid


def knn_classifier (train_grid, test_grid)

	#Set index as row id for test_grid
	row_id = test_grid.index 	
	#Classifier knn
	le = LabelEncoder()
    	y = le.fit_transform(train_grid.place_id.values)
	#Delete place_id in grid
	del train_grid ['place_id']
	clf = KNeighborsClassifier(n_neighbors=25, weights='distance', 
                               metric='manhattan')
    	clf.fit(train_grid, y)
    	y_pred = clf.predict_proba(test_grid)
    	pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])    
    	return pred_labels, row_id

	
	#Main program
if __name__ == '__main__':
	print ('Starting program')
	x_cord1 = 1
	x_cord2 = 1.25
	y_cord1 = 2.5
	y_cord2 = 2.75
	print ('Enter input file name')
	test_csv = 'test_grid.csv'
	print ('Input file name is %s' %input_csv)
	train_csv = 'train_grid.csv'
	train_grid = data_prep (test_csv,x_cord1,x_cord2,y_cord1,y_cord2)
	test_grid = data_prep (train_csv,x_cord1,x_cord2,y_cord1,y_cord2)
	new_train = train_prep (train_grid)
	pred_labels,row_id = knn_classifier (new_train,test_grid)
	print pred_labels row_id
	 
	




 


