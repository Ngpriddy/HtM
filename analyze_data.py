"""
Script to analyze HackTheMachine data. No particular purpose at the moment.

author: Zachary Davis
date: 08/29/2019
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Data loading stuff; pickle datasets for faster access
load_pickle = True 

if load_pickle:
  all_maf_data = pd.read_pickle("pkl_files/MAF.pkl")
  all_msp_data = pd.read_pickle("pkl_files/MSP.pkl")
else:
  all_maf_data = pd.read_csv("data/MAF.csv")
  all_maf_data.to_pickle("pkl_files/MAF.pkl")
  all_msp_data = pd.read_csv("data/MSP.csv")
  all_msp_data.to_pickle("pkl_files/MSP.pkl")

# Print column names for my own reference 
for col in all_maf_data.columns:
  print(col)

# Remove nan rows where all entries are 'nan'
all_maf_data = all_maf_data.dropna(how='all')

#Replace 'Yes' entries with 1 in binary-valued columns and 'NaN' entires with 0
col_num = 0
for col in all_maf_data.columns:
  if col_num > 8:
    all_maf_data[col] = all_maf_data[col].map({'Yes': 1}).fillna(0.0)
  col_num += 1

"""
# Compute pairwise Pearson correlations between every binary column and print
# the result
# Don't you love when you do something the hard way and find out there is a 
# built-in function that does the same thing?
seq_1_col = 0 
for col in all_maf_data.columns:
  if seq_1_col > 8:
    for i in range(seq_1_col, 16):
      seq_1 = all_maf_data.iloc[:,seq_1_col]
      seq_2 = all_maf_data.iloc[:,i]
      corr = pearsonr(seq_1, seq_2)    
      print("Pearson correlation for {0} and {1}: {2}".format(all_maf_data.columns[seq_1_col], all_maf_data.columns[i], corr[0]))
  seq_1_col += 1
"""
# Compute pairwise Pearson correlations the easy way, plot
corr_matrix = all_maf_data[all_maf_data.columns[8:]].corr(method='pearson')

plt.matshow(corr_matrix)
plt.xticks(range(corr_matrix.shape[1]), corr_matrix.columns, rotation=45, fontsize = 6)
plt.yticks(range(corr_matrix.shape[1]), corr_matrix.columns)
cb = plt.colorbar()
plt.show()
