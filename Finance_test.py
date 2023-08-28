import pandas as pd
import yfinance as yf
import datetime

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#Fixing intraday date as 10th August 2023 and ADR opening and closing time as 9:30am and 4pm respectively
start_time = datetime.datetime(2023, 8, 10, 9, 30)
end_time = datetime.datetime(2023, 8, 10, 16, 0)
#Extracting ADR data for Adidas with 1 minute intervals
adr_df = yf.download('ADDYY',start=start_time,end=end_time,interval='1m',progress=False)
#Extracting European stocks data for Adidas
edr_df = yf.download('ADS.DE',start=start_time,end=end_time,interval='1m',progress=False)

#Returns for ADR
adr_df['ADR_Return'] = (adr_df['Close'] - adr_df['Open']) / adr_df['Open']

#Returns for European stock
edr_df['European_Stock_Return'] = (edr_df['Close'] - edr_df['Open']) / edr_df['Open']

#Extracting intraday data for SP500 index
sp500_df = yf.download('^GSPC',start=start_time, end=end_time, interval='1m', progress=False)

#Returns for SP500
sp500_df['SP500_Return'] = (sp500_df['Close'] - sp500_df['Open']) / sp500_df['Open']

#Combining SP500 returns with ADR and European stock returns
combined_df = adr_df.merge(edr_df[['European_Stock_Return']], left_index=True, right_index=True)
combined_df = combined_df.merge(sp500_df[['SP500_Return']], left_index=True, right_index=True)

#Residual intraday ADR returns
combined_df['Residual_ADR_Return'] = combined_df['ADR_Return'] - combined_df['SP500_Return']

#Fixing the European Market Close time as 1PM
european_market_close_time = datetime.time(13, 0)

#Filtering intraday data to include the time period from European Market Close to US Market Close
filtered_combined_df = combined_df[combined_df.index.time >= european_market_close_time]

#ADR return and European stock return for the filtered time period
filtered_combined_df['Filtered_ADR_Return'] = (filtered_combined_df['Close'] - filtered_combined_df['Open']) / filtered_combined_df['Open']
filtered_combined_df['Filtered_European_Stock_Return'] = (filtered_combined_df['Close'] - filtered_combined_df['Open']) / filtered_combined_df['Open']

#Residual intraday ADR returns for the filtered time period
filtered_combined_df['Residual_Intraday_ADR_Return'] = filtered_combined_df['Filtered_ADR_Return'] - filtered_combined_df['SP500_Return']

average_adr_return = filtered_combined_df['Filtered_ADR_Return'].mean()

#Abnormality measure
filtered_combined_df['Abnormality_Measure'] = filtered_combined_df['Filtered_ADR_Return']- average_adr_return

#Define positive threshold and negative threshold
positive_threshold = 0.000001
negative_threshold = -0.0000001

# Categorize events based on abnormality measure and thresholds
filtered_combined_df['Event_Type'] = 'Neutral'
filtered_combined_df.loc[filtered_combined_df['Abnormality_Measure'] > positive_threshold, 'Event_Type'] = 'Positive'
filtered_combined_df.loc[filtered_combined_df['Abnormality_Measure'] < negative_threshold, 'Event_Type'] = 'Negative'

print(filtered_combined_df.head())

import matplotlib.pyplot as plt

#Cumulative specific returns for each stock
filtered_combined_df['Filtered_ADR_Cumulative_Return'] = (1 + filtered_combined_df['Residual_Intraday_ADR_Return']).cumprod()
filtered_combined_df['Filtered_European_Stock_Cumulative_Return'] = (1 + filtered_combined_df['Filtered_European_Stock_Return']).cumprod()

#Grouping data
event_groups = filtered_combined_df.groupby('Event_Type')

#Expected conditional cumulative returns for each event type
expected_cumulative_returns = event_groups[['Filtered_ADR_Cumulative_Return', 'Filtered_European_Stock_Cumulative_Return']].mean()

print(expected_cumulative_returns)

#Plot
x_values = [(time.hour * 60 + time.minute) for time in filtered_combined_df.index.time]

plt.figure(figsize=(10, 6))
for event_type in expected_cumulative_returns.index:
    plt.plot(x_values,
             [expected_cumulative_returns.loc[event_type, 'Filtered_ADR_Cumulative_Return']] * len(filtered_combined_df),
             label=f'ADR ({event_type})')
    plt.plot(x_values,
             [expected_cumulative_returns.loc[event_type, 'Filtered_European_Stock_Cumulative_Return']] * len(filtered_combined_df),
             label=f'European Stock ({event_type})')

plt.xlabel('Elapsed Time (in minutes)')
plt.ylabel('Expected Cumulative Return')
plt.title('Expected Conditional Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()


