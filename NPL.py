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
df = pd.read_csv(response, header=0 )

# clean data:

#   1. remove null variables (should not have much, might not be worth using averages)

#   2.2 - make categorical data numerical


df['Gender_client_num'] = 0   #create new dataframe 

#map it to the gender dataframe, with 0 for F and 1 for M
df['Gender_client_num'] = \
	df['Gender (Client)'].map( {'female': 0, 'male': 1}).astype(int)

#print df

# we need to somehow map this over the dataframes that arent INTS
"""def dollars_dec(x):
	if type(x) == 'object':
		return Decimal(x.strip('$'))
	else:
	 return x"""

"""def cleanup(item): 
	return float(str(item).replace("$", ""))"""

# drop the random generator 
df = df.drop(['Random stuff'], axis=1)

# drop the male and female column 
df = df.drop(['Gender (Client)'], axis=1)


headers = list(df.columns.values)

print 'headers = %s' % headers



for header in df[headers]: 
	print header[1] 

	print 'header = %s is a %s' % (header,type(df[header]))

#print df['Unrecovered']

# make a new dataframe with the strings 
"""df2 = pd.DataFrame() 
for header in headers:
	if df[header].dtypes == 'object':
		df2.append(df[header])"""
#print df2

# clean the strings and make them floats 
#remove the dollar signs 

#print df2

"""def dollaraway(x): 
	if type(x)=='object':
		print 'yay! object'
	else: 
		print 'uuuh, floaty!'   """



#print df.dtypes[df.dtypes.map(lambda x: x=='object')]

#df2 = df.applymap(dollars_dec)

#print df2

#   3. Convert numbers into decimal format, and normalize data 

#   4. Drop non-necessary data

# convert data to numpy array 

# separate data into training and test sets 

# Create the random forest object which will include all the parameters for the fit

# Fit the training data to the label and create the decision trees

# Test the decision trees and run it on the test data
