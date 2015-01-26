# import libraries 
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pylab as P
import csv
import urllib2

# import sample data from S3 
url = 'https://trello-attachments.s3.amazonaws.com/54522d0bd5a9e7596679dd06/54c620fa57b111af46716e5c/a79bc405e58353ef79e2bbbf9adc3290/Mock_data_LGD_-_Sheet1.csv'
response = urllib2.urlopen(url)

# convert csv data to pandas dataframe 
df = pd.read_csv(response, header=0)

# clean data:

#   1. remove null variables (should not have much, might not be worth using averages)

#   2. make categorical data numerical 
#   2.1 - separate categorical and numerical data
cat_data = []
num_data = []
for column in df:
  if #there is a non numerical entry in the column:
    #add column to the cat_columns dataframe
  else:
    #add column to the num columns dataframe

#   2.2 - make categorical data numerical

#   3. Normalize data 

#   4. Drop non-necessary data

# convert data to numpy array 

# separate data into training and test sets 

# Create the random forest object which will include all the parameters for the fit

# Fit the training data to the label and create the decision trees

# Test the decision trees and run it on the test data
