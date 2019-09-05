"""
Script to compare distributions of MSP codes before, during, and after corrosion-related maintenace actions

Author: Zachary Davis
Date: 2019-09-04
"""
import pandas as pd
from utils import date_time_convert
import matplotlib.pyplot as plt

maf_data = pd.read_csv('data/converted_MAF.csv')
maf_data = maf_data.sort_values(by=['Received Date'])
msp_data = pd.read_csv('data/converted_MSP.csv')
msp_data = msp_data.sort_values(by=['ZULU_TIME'])

# Get MSP codes during flight period of 2012-01-03 --> 2012-01-26
prior_flight_msp = msp_data[(msp_data['ZULU_TIME'] <= '2012-01-26') & (msp_data['ZULU_TIME'] >= '2012-01-03') & (msp_data['FLIGHT_MODE'] == ' InFlight')]
prior_flight_msp_codes = prior_flight_msp['MSP'].value_counts()
print("MSP codes before maintenance: {0}".format(prior_flight_msp_codes))
prior_flight_msp_codes.to_csv("prior.csv")

# Get MSP codes during maintenance period of 2012-01-26 --> 2012-02-12
maintenance_msp = msp_data[(msp_data['ZULU_TIME'] <= '2012-02-12') & (msp_data['ZULU_TIME'] >= '2012-01-26') & (msp_data['FLIGHT_MODE'] == ' EngineTurn')]
maintenance_msp_codes = maintenance_msp['MSP'].value_counts()
print("MSP codes during maintenance: {0}".format(maintenance_msp_codes))
maintenance_msp_codes.to_csv("maintenance.csv")

# Get MSP codes during flight period of 2012-02-12 --> 2012-03-06
post_flight_msp = msp_data[(msp_data['ZULU_TIME'] <= '2012-03-06') & (msp_data['ZULU_TIME'] >= '2012-02-12') & (msp_data['FLIGHT_MODE'] == ' InFlight')]
post_flight_msp_codes = post_flight_msp['MSP'].value_counts()
print("MSP codes after maintenance: {0}".format(post_flight_msp_codes))
post_flight_msp_codes.to_csv("post.csv")
