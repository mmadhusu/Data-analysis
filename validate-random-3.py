#!/usr/bin/python
import sys
import csv
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from calculate_accuracy import average_score
#*****************************************THIS WORKS LIKE A CHARM*****************************************#
def data_prep(input_csv,x_cord1,x_cord2,y_cord1,y_cord2):
	
        #Read the csv file
	grid= pd.read_csv(input_csv, index_col = 'row_id')
  
        #Create data grid
        grid = grid [(grid['x'] > x_cord1) & (grid ['x'] < x_cord2) & (grid['y'] > y_cord1) & (grid['y'] < y_cord2)]
	#Feature engineering : Creating the features
	grid ['Hour'] = (grid ['time']/60)%24
	grid ['Weekday'] = (grid ['time']/(60*24))%7
	grid ['Month'] = (grid ['time']/(60*24*30))%12
	grid ['Year'] = grid ['time']/(60*24*365)
	grid ['Day'] = grid ['time']/(60*24) % 365
	grid.to_csv ('test-grid.csv')
	#print grid

	#time provides the same info. Delete the time variable
        del grid['time']

def feature_engg (input_csv):
	#Read the csv file
        grid= pd.read_csv(input_csv, index_col = 'row_id')

	#Feature Engineering : Assigning the weights for each feature
        feature_weights =  [125, 225, 10 , 1, 1, 1, 1]
        grid ['x'] = grid.x.values * feature_weights[0]
        grid ['y'] = grid.y.values * feature_weights[1]
        grid['Hour'] = grid.Hour.values * feature_weights[2]
        grid['Weekday'] = grid.Weekday.values * feature_weights[3]
        grid['Day'] = (grid.Day * feature_weights[4]).astype(int)
        grid['Month'] = grid.Month * feature_weights[5]
        grid['Year'] = grid.Year * feature_weights[6]

	return grid

def create_validation_set(df):
	mask  = np.random.rand(len(df)) < 0.3 #Random sampling in ~ 70 30 ratio
	validate = df[mask]
	#print validate.describe()
	train =  df[~mask]
	#print train.describe()
	#Divide the validate into test and validate tp create a hold-out test
	mask2 = np.random.rand(len(validate)) < 0.5
	valid = validate [mask2]
	test   = validate [~mask2] 
	valid.to_csv ('validate-random-sampling.csv')
	test.to_csv ('test-random-sampling.csv')
	return valid, train, test

	
def train_prep (train_grid):	
	#Keep the cutoff of place id values as k=3
	#Any other place_id values are just dropped from the grid
	cut_off = 3
	placeid_count = train_grid.place_id.value_counts() #Counts the number of unique occurences of a place id
	flag = (placeid_count[train_grid.place_id.values] > cut_off).values # Adds boolean if the number of values > cutoff
	modified_train_grid = train_grid.loc[flag] #Considers only the place_ids that occur more than the cutoff times
	return modified_train_grid


def knn_classifier (train_grid, validate):

	#Set index as row id for test_grid
	row_id = validate.index 	
	#Classifier knn
	le = LabelEncoder()
        place_ids = validate['place_id'].tolist()
	del validate['place_id']
    	y = le.fit_transform(train_grid.place_id.values)
	#Delete place_id in grid for
	del train_grid ['place_id']
	clf = KNeighborsClassifier(n_neighbors=15, weights='distance', 
                               metric='manhattan')
    	clf.fit(train_grid, y)
    	y_pred = clf.predict_proba(validate)
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
	print ('Starting program')
	#x_cord1 = 1
	#x_cord2 = 9.75
	#y_cord1 = 1.5
	#y_cord2 = 9.75
	#train_csv = 'train-manip.csv'
	#train_grid = data_prep (train_csv,x_cord1,x_cord2,y_cord1,y_cord2)
	#new_train = train_prep (train_grid)
	new_train = pd.read_csv ('train-grid.csv', index_col = 'row_id')
	#Create validation set
	validate, train, test = create_validation_set(new_train)
	train.to_csv ('random-3-train.csv')
	train = feature_engg('random-3-train.csv')
	print len(validate)
	pred_labels,row_id, place_ids = knn_classifier (train, test)
	#print pred_labels
	#print row_id 
	#print place_ids
	#Converting a list into a list of lists to call the average_score
	list_of_lists = []
	for a in place_ids:
		list_of_lists.append([a])
	#print list_of_lists
	#print len(list_of_lists)
	#print pred_labels[0]
	create_results_file (pred_labels)
	#print row_id
	average_score(list_of_lists, pred_labels)
	#If it were single prediction, how many as the correct prediction
	first_prediction = pred_labels[:,0]
	first_prediction_list = []
	for a in first_prediction:
		first_prediction_list.append([a])
	average_score (list_of_lists, first_prediction_list)
	 
	




 


