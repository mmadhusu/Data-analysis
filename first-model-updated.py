#!/usr/bin/python
import sys
import csv
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from calculate_accuracy import average_score
import pdb

def data_prep(input_csv,x_cord1,x_cord2,y_cord1,y_cord2):
	
        #Read the csv file
	grid = pd.read_csv(input_csv, index_col = 'row_id')
        #Create data grid
	print ("Creating data grid")
        grid = grid[(grid['x'] > x_cord1) & (grid['x'] < x_cord2) & (grid['y'] > y_cord1) & (grid['y'] < y_cord2)]
	print ("Feature Engineering")
	#Feature engineering : Creating the features
	grid['Hour'] = (grid.loc[:,'time']/60)%24
	grid['Weekday'] = (grid['time']/(60*24))%7
	grid['Month'] = (grid['time']/(60*24*30))%12
	grid['Year'] = grid['time']/(60*24*365)
	grid['Day'] = grid['time']/(60*24) % 365
	grid.to_csv ('test-grid.csv')

	#time provides the same info. Delete the time variable
        del grid['time']

	#Feature Engineering : Assigning the weights for each feature
        feature_weights =  [500, 1000, 4, 3, 1, 2, 10]
        grid['x'] = grid ['x'] * feature_weights[0]
        grid['y'] = grid.y.values * feature_weights[1]
        grid['Hour'] = grid.Hour.values * feature_weights[2]
        grid['Weekday'] = grid.Weekday.values * feature_weights[3]
        grid['Day'] = (grid.Day * feature_weights[4]).astype(int)
        grid['Month'] = grid.Month * feature_weights[5]
        grid['Year'] = grid.Year * feature_weights[6]

	return grid


def train_prep (train_grid, cut_off):	
	#Any other place_id values are just dropped from the grid
	placeid_count = train_grid.place_id.value_counts() #Counts the number of unique occurences of a place id
	flag = (placeid_count[train_grid.place_id.values] > cut_off).values # Adds boolean if the number of values > cutoff
	modified_train_grid = train_grid.loc[flag] #Considers only the place_ids that occur more than the cutoff times
	return modified_train_grid


def knn_classifier (train_grid, test_grid, k):

	#Set index as row id for test_grid
	row_id = test_grid.index 	
	#Store place_ids as a list in order to calculate MAP3
	place_ids = train_grid['place_id'].tolist()
        #Classifier knn
	le = LabelEncoder()
    	y = le.fit_transform(train_grid.place_id.values)
	#Delete place_id in grid
	del train_grid['place_id']
	clf = KNeighborsClassifier(n_neighbors=10, weights='distance', 
                               metric='manhattan')
    	clf.fit(train_grid, y)
	if k == "1":
    	  y_pred = clf.predict_proba(test_grid)
	else:
	  y_pred = clf.predict_proba(train_grid)

    	pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])    
    	return pred_labels, row_id, place_ids

def create_results_file(preds):
	
	df_aux = pd.DataFrame (preds, dtype=str, columns = ['l1', 'l2', 'l3'])
	ds_sub = df_aux.l1.str.cat ([df_aux.l2, df_aux.l3], sep=' ')

	#Writing to a csv file
	ds_sub.name = 'place_id'
	ds_sub.to_csv ('sub_results.csv', index=True, header =True, index_label= 'row_id') 

#def calculate_mean_average_precision ()
	
				
	#Main program
if __name__ == '__main__':
	print ('******************Starting program***************************')
	#Create the grid with the following co-ordinates
	x_cord1 = 1
	x_cord2 = 3.25
	y_cord1 = 1
	y_cord2 = 3.25
	# Names of the train and test data sets
	test_csv = 'test.csv'
	train_csv = 'train.csv'
	#Create the grid and the weights for each feature
        train_grid = data_prep (train_csv,x_cord1,x_cord2,y_cord1,y_cord2)
	test_grid = data_prep (test_csv,x_cord1,x_cord2,y_cord1,y_cord2)
	print ('***************Printing test grid**********************')
	print test_grid
	#Take only place_ids which occur more than cut_off times
	new_train = train_prep (train_grid,0)
	train_copy = new_train.copy()
	#Apply classifier to predict place_ids of test data set
	pred_labels,row_id,place_ids = knn_classifier (new_train,test_grid,1)
	#Create results file
	create_results_file (pred_labels)
	#Apply classifier onto training set to calculate accuracy
	pred_labels, row_id, place_ids = knn_classifier (train_copy, train_copy,2)
	print pred_labels
	print row_id
	#Change the format to list of lists to call average precision score
	list_of_lists = []
        for a in place_ids:
                list_of_lists.append([a])
	average_score(list_of_lists, pred_labels)
	print ('*******************End of program********************************')
	
	 
	




 


