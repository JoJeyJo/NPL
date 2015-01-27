#Random stuff import libraries 

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pylab as P
import csv
import urllib2
from decimal import Decimal

# import sample data from S3 
url = 'https://trello-attachments.s3.amazonaws.com/54522d0bd5a9e7596679dd06/54c620fa57b111af46716e5c/a79bc405e58353ef79e2bbbf9adc3290/Mock_data_LGD_-_Sheet1.csv'
response = urllib2.urlopen(url)

# convert csv data to pandas dataframe 

loans = pd.read_csv(response)


# ---------------------------------------------------------#
#                Drop rows with null variables             #
# ---------------------------------------------------------#

loans = loans.dropna()

# ---------------------------------------------------------#
#                Replace currency with Decimal             #
# ---------------------------------------------------------#

# define what being currency means: 
def is_currency(serie):
	if loans[serie].dtypes == 'object':
		if loans[serie].iloc[1][0] == '$': #if it has a $ on its first line
			return True
	else: 
		return False

# Get a list of index names: 
loan_parameters = list(loans.columns.values)

for parameter in loan_parameters:
	if is_currency(parameter):
		loans[parameter] = loans[parameter].map(lambda x: Decimal(x.replace('$', '').replace(',','')))
		
#print loans.head(5)


# ---------------------------------------------------------#
#                Make categorical data numerical           #
# ---------------------------------------------------------#


#define what being categorical means: 
def is_categorical(serie):
	first = loans[serie].iloc[1]  #first line
	#if the first line is a string, then true 
	if  isinstance(first, str):
		return True
	else: 
		return False

def map_uniques(serie):
	# we want to build a dictionary of {unique0: 0, unique1 : 1, ... }
	uniques = loans[serie].unique()
	num_map = {}
	index = 0 
	for unique in uniques:
		num_map[unique]= [index]
		index += 1 
	return num_map

def map_series(serie):
	loans[serie] = loans[serie].map(map_uniques(serie)) 
#	print loans[serie]

"""df['Gender_client_num'] = \
	df['Gender (Client)'].map( {'female': 0, 'male': 1}).astype(int)"""


a = 'Gender (Client)'
print map_series(a)
print loans[a].iloc[1] # should be a 0
print type(loans[a].iloc[1])  # should be an float

#for parameter in loan_parameters:
#	if is_categorical(parameter):
#		map_uniques(parameter)


# 	2.2.1 - Separate numerical and categorical data 


# map new column to the gender dataframe, with 0 for F and 1 for M

# we need to somehow map this over the dataframes that arent INTS

# drop the random generator 

# drop the male and female column 

#   3. Convert numbers into decimal format, and normalize data 

#   4. Drop non-necessary data

# convert data to numpy array 

# separate data into training and test sets 

# Create the random forest object which will include all the parameters for the fit

# Fit the training data to the label and create the decision trees

# Test the decision trees and run it on the test data	
