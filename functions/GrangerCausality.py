from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

def apply_differencing(series,max_diff,alpha):
    # H0: Time-Series is Non-Stationary
    # H1: Time-Series is Stationary
    
    #if p_value<= alpha: #reject null hypothesis (stationary)
    #else: # fail to reject null hypothesis (non-stationary)
    
    num_diff=0
    result = adfuller(series.dropna(),autolag='AIC')
    p_value = result[1]
    
    while p_value > alpha and num_diff < max_diff: #while non-stationary, keep differencing the series to make it stationary
        series = series.diff().dropna()
        result =adfuller(series,autolag='AIC')
        p_value = result[1]
        num_diff +=1
    
    return num_diff,p_value

def make_series_stationary(df, max_diff=12, alpha=0.05):
    stationary_info = {}  # Dictionary to store information about stationary transformation for each variable
    
    # Loop through each column (except the datetime column) in the DataFrame
    for col in df.columns[1:]:
        # Apply differencing to the column
        num_diff, p_value = apply_differencing(df[col], max_diff, alpha)
        
        # Store information about stationary transformation for the variable
        stationary_info[col] = {'num_diff': num_diff, 'p_value': p_value}
        
    return stationary_info


#We reject the null hypothesis
#that x2 does not Granger cause x1 if the pvalues are below a desired size
#of the test.

def differencing(series, num_diff):
    for _ in range(num_diff):
        series = series.diff().dropna()
    return series

def make_differenced_dataframe(df, stationary_info):
    new_df = pd.DataFrame(index=df.index)
    
    for col, info in stationary_info.items():
        if info['num_diff'] > 0:
            new_df[col] = differencing(df[col], info['num_diff'])
        else:
            new_df[col] = df[col]
    
    return new_df

def granger_causality_test(df, target_variable, max_lags=9,test='ssr_chi2test'):
    results = {}
    
    for col in df.columns:
        if col != target_variable:
            data = df[[target_variable, col]].dropna()
            best_p_value = 1.0  # Initialize with a large p-value
            best_lag = None
            for lag in range(1, max_lags + 1):
                granger_result = grangercausalitytests(data, maxlag=lag)
                chi2_p_value = granger_result[lag][0][test][1]
                if chi2_p_value < best_p_value:
                    best_p_value = chi2_p_value
                    best_lag = lag
            results[col] = {'p_value': best_p_value, 'best_lag': best_lag}
                
    return results