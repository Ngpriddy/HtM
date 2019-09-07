"""
Testing date/time convert function
"""
import pandas as pd
from utils import date_time_convert
msp_data = pd.read_csv("data/converted_msp.csv")
maf_data = pd.read_csv("data/converted_maf.csv")
AIRCRAFT = 41
msp_data = msp_data.sort_values(by='ZULU_TIME')
maf_data = maf_data.sort_values(by='Received Date')

msp_data_single_aircraft = msp_data[msp_data['AIRCRAFT'] == 1]
#maf_data_single_aircraft = maf_data[(maf_data['Aircraft'] == 1.0) & (maf_data['Corrosion'] == 'Yes')]
maf_data_single_aircraft = maf_data[(maf_data['Aircraft'] == 1.0) & (maf_data['Failure'] == 'Yes')]

maf_data_single_aircraft = maf_data_single_aircraft.drop(columns=['Routine Maintenance', 'Unscheduled Maintenance', 'Job Code', 'Aircraft',
    'Bare Metal', 'Transaction Code', 'Malfunction Code', 'Description of Problem', 'Action Taken Code', 'Correction of Problem'])
"""
#print(msp_data_single_aircraft[msp_data_single_aircraft['ZULU_TIME'] >= '2014-11-17'])
print(maf_data_single_aircraft)
#print(maf_data_single_aircraft[maf_data_single_aircraft['Received Date'] >= '2014-09-24'])

msp_pre_maintenance = msp_data_single_aircraft[(msp_data_single_aircraft['ZULU_TIME'] <= '2014-09-18') & (msp_data_single_aircraft['ZULU_TIME'] >=
    '2014-09-04') & (msp_data_single_aircraft['FLIGHT_MODE'] == ' InFlight')]

msp_during_maintenance = msp_data_single_aircraft[(msp_data_single_aircraft['ZULU_TIME'] >= '2014-09-18') & (msp_data_single_aircraft['ZULU_TIME'] <
    '2014-09-19') & (msp_data_single_aircraft['FLIGHT_MODE'] == ' EngineTurn')]

msp_post_maintenance = msp_data_single_aircraft[(msp_data_single_aircraft['ZULU_TIME'] >= '2014-09-19') & (msp_data_single_aircraft['ZULU_TIME'] <=
    '2014-10-03') & (msp_data_single_aircraft['FLIGHT_MODE'] == ' InFlight')]

print("MSP codes one month prior to maintenance: \n {0}".format(msp_pre_maintenance['MSP'].value_counts()))
print("MSP codes during maintenance period: \n {0}".format(msp_during_maintenance['MSP'].value_counts()))
print("MSP codes one month after maintenance: \n {0}".format(msp_post_maintenance['MSP'].value_counts()))
"""
#for col in maf_data.columns:
#    print(col)

# Create column to determine if 'wiring inspection' is mentioned in problem description
maf_data['wire mention'] = maf_data['Description of Problem'].str.find('wiring inspection')

# Create dataframe of only MAFs mentioning wire inspections
wire_data = maf_data[(maf_data['wire mention'] != -1)]
wire_inspection_dates = wire_data.iloc[:,9].values
print(wire_inspection_dates)

msp_df = pd.DataFrame()
for i in range(wire_data.shape[0]):
    # Determine the MSP codes that occured between the 'start' and 'end' times of the MAFs that mention wiring inspections
    msp_during_maintenance = msp_data[(msp_data['ZULU_TIME'] >= wire_data.iloc[i,8]) & (msp_data['ZULU_TIME'] < wire_data.iloc[i,9]) & (msp_data['FLIGHT_MODE']
        == ' EngineTurn') & (msp_data['AIRCRAFT'] == wire_data.iloc[i,2])]
    # Add MSP codes to running list    
    msp_df = pd.concat([msp_df, msp_during_maintenance['MSP'].value_counts().reset_index().rename(columns={'index': 'MSP', '0': 'Count'})])

# Collapse repeat codes into one row for total of all MSP codes that occured across all aircraft during wiring inspections
msp_df.columns = ['MSP', 'counts']
aggregation_functions = {'counts': 'sum'}
msp_collapsed = msp_df.groupby('MSP', as_index=False).aggregate(aggregation_functions).reindex(columns=msp_df.columns)

print(msp_df)
print(msp_collapsed)
msp_collapsed.to_csv("msp_codes_during_wiring_inspection.csv")
