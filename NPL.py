#Import libraries 
import pandas as pd
import numpy as np
import pylab as P
import csv
import matplotlib.pyplot as plt
import urllib2
from sklearn.ensemble import RandomForestRegressor
from decimal import Decimal
import random

# ---------------------------------------------------------#
#                           Get data                       #
# ---------------------------------------------------------#

print('importing data...')
# import sample data from S3 
url = 'https://trello-attachments.s3.amazonaws.com/54522d0bd5a9e7596679dd06/54c620fa57b111af46716e5c/a79bc405e58353ef79e2bbbf9adc3290/Mock_data_LGD_-_Sheet1.csv'
response = urllib2.urlopen(url)

# convert csv data to pandas dataframe 
loans = pd.read_csv(response)

df = loans

# ---------------------------------------------------------#
#                      Get rid of nulls                    #
# ---------------------------------------------------------#

print('cleaning data...')

# Drop rows with null variables
df = df.dropna()

# ---------------------------------------------------------#
#                Replace currency with Decimal             #
# ---------------------------------------------------------#

# define what being currency means: 
def is_currency(serie):
	if df[serie].dtypes == 'object':
		if df[serie].iloc[1][0] == '$': #if it has a $ on its first line
			return True
	else: 
		return False

# Get a list of index names: 
loan_parameters = list(df.columns.values)

for parameter in loan_parameters:
	if is_currency(parameter):
		df[parameter] = df[parameter].map(lambda x: float(x.replace('$', '').replace(',','')))
		# these should not be floats, for greater accuracy they should be Decimal

# ---------------------------------------------------------#
#                 Make categorical data numerical          #
# ---------------------------------------------------------#

#define what being categorical means: 
def is_categorical(serie):
	first = df[serie].iloc[1]  #first line
	#if the first line is a string, then true 
	if  isinstance(first, str):
		return True
	else: 
		return False

# make a map of unique values and corresponding numerical values: 
def map_uniques(serie):
	# we want to build a dictionary of {unique0: 0, unique1 : 1, ... }
	uniques = df[serie].unique()
	num_map = {}
	index = 0 
	for unique in uniques:
		num_map[unique]= index
		index += 1 
	return num_map

# map the unique values to the numerical values: 
def map_series(serie):
	df[serie] = df[serie].map(map_uniques(serie)) 

# Turn all categorical parameters into numerical parameters  
for parameter in loan_parameters:
	if is_categorical(parameter):
		map_series(parameter)


# ---------------------------------------------------------#
#                   Drop unnecessary columns               #
#           (might come from user interface later)         #
#          (Should also drop all the Object dtypes)        #
# ---------------------------------------------------------#

# drop the random generator 
df = df.drop(['Random stuff'], axis = 1) # axis 1 means column


# ---------------------------------------------------------#
#             Split into test and training data            #
# ---------------------------------------------------------#

# separate data into training and test sets 
def split_dataframe(dataframe): 

	# initialize empty training and test sets 
	full_set = dataframe.values
	training_set = []
	test_set = []
	num_rows = df.shape[0]
	percent_test = 0.3

	#randomly assign lines to either trainig or test
	for line in range(0,num_rows):
		i = random.random()
		if i < percent_test:
			training_set.append(full_set[line])
			#print(full_set[line])
		else: 
			test_set.append(full_set[line])
	return test_set, training_set

#create test and training set
test_set, training_set = split_dataframe(df)

# ---------------------------------------------------------#
#                     Prepare training sets                #
# ---------------------------------------------------------#

def split_data_and_target(set): 
	num_cols = len(set[0])-1
	data = []
	target = []
	for line in set:
 		data.append(line[0:num_cols])
		target.append(line[num_cols])
	return data, target 

# ---------------------------------------------------------#
#                     Generate forest                      #
# ---------------------------------------------------------#

# random forest code
rf = RandomForestRegressor(n_estimators=150, min_samples_split=2, n_jobs=-1)

#train the model
def train_model(training_set):
	print('fitting the model...')	
	train, target = split_data_and_target (training_set)
	predictor = rf.fit(train, target)
	return predictor

#use the model to make  predictions
def predict(test_set):
	print 'making predictions...'
	train, target = split_data_and_target (test_set)
	prediction = rf.predict(train)
	return prediction

train_model(training_set)
predictions = predict(test_set)

# ---------------------------------------------------------#
#                   Performance metrics                    #
# ---------------------------------------------------------#

for i in range(0,5):
	print test_set[i]
	target_index = len(test_set[0])-1
	actual = test_set[i][target_index]
	predicted = predictions[i]
	print 'Predicted: %1.0f' % predicted
	print 'Correct: %1.0f' % actual
	print ''

def measure_error(predictions, test_set):

	data_set, target = split_data_and_target(test_set)
	set_size = len(predictions)-1
	error_list = []

	for i in range(0,set_size):
		error = abs(predictions[i] - target[i]) 
		error_list.append(error) 

	#plt.plot(predictions, target, 'ro')
	#plt.show()

	#plot the sorted list of the dimention of the errors
	#plt.plot(sorted(error_list), 'ro')
	#plt.show()
	#return error_list

error_data = measure_error(predictions,test_set)

# ---------------------------------------------------------#
#                      Feature importance                  #
# ---------------------------------------------------------#

feat_imp = rf.feature_importances_

print feat_imp
print loan_parameters 

def graph_feat_importance(data, labels):

	a = data # y values 
	b = labels # x values
	d = []    # make the x values numerical
	for i in range (0,len(b)):
		d.append(i)

	plt.bar(d,a, color='c', align='center')
	plt.title('Feature importance')
	plt.ylabel('Relative Importance')
	plt.xlabel('Feature')
	plt.xticks(d,b)
	plt.show()


graph_feat_importance(feat_imp, loan_parameters)


#print 'the mean accuracy was %1.0f' % np.mean(accuracy_data)
#print 'the median accuracy was %1.0f' % np.median(accuracy_data)
