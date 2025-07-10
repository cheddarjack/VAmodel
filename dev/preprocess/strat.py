import numpy as np
import pandas as pd
import time
from numba import njit
import itertools

# -------------------------------------------------------------------    
# ----------------------- Helper Functions --------------------------
# -------------------------------------------------------------------

def calculate_complex_value(average_size, low_average, low_value, high_average, high_value, threshold, min_value, max_value):
    
    if average_size <= 0:
        complex_value = max_value  # Starting value at average_size = 0
    elif 0 < average_size <= low_average:
        m_initial = (low_value - max_value) / (low_average - 0)
        b_initial = max_value  # At average_size = 0, complex_value = max_value
        complex_value = m_initial * average_size + b_initial
    elif low_average < average_size <= high_average:
        m_middle = (high_value - low_value) / (high_average - low_average)
        b_middle = low_value - m_middle * low_average
        complex_value = m_middle * average_size + b_middle
    elif high_average < average_size < threshold:
        m_decay = (min_value - high_value) / (threshold - high_average)
        b_decay = high_value - m_decay * high_average
        complex_value = m_decay * average_size + b_decay
    else:
        complex_value = min_value
    return complex_value

def stochastic(length, tick_data):
    
    ohlc_bars = tick_data.groupby('bar_id').agg(
        High=('Last', 'max'),
        Low=('Last', 'min'),
    ).reset_index().astype({'bar_id': 'uint32'})

    ohlc_bars['highest_high'] = (ohlc_bars['High'].rolling(window=(length - 1), min_periods=1).max().shift()).astype('float32').round(2)
    ohlc_bars['lowest_low'] = (ohlc_bars['Low'].rolling(window=(length - 1), min_periods=1).min().shift()).astype('float32').round(2)

    ohlc_bars.set_index('bar_id', inplace=True)
    tick_data['prev_highest_high'] = tick_data['bar_id'].map(ohlc_bars['highest_high']).astype('float32').round(2)
    tick_data['prev_lowest_low'] = tick_data['bar_id'].map(ohlc_bars['lowest_low']).astype('float32').round(2)

    tick_data['highest_high'] = np.maximum(tick_data['prev_highest_high'], tick_data['High']).astype('float32').round(2)
    tick_data['lowest_low'] = np.minimum(tick_data['prev_lowest_low'], tick_data['Low']).astype('float32').round(2)

    price_range = tick_data['highest_high'] - tick_data['lowest_low']
    Fast_K = np.where(
        price_range != 0,
        (tick_data['Last'] - tick_data['lowest_low']) / price_range * 100,
        50  # Default value if no price movement
    )
    # drop columns 
    cols_to_drop = ['prev_highest_high', 'prev_lowest_low', 'highest_high', 'lowest_low']
    tick_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return Fast_K

def SMA(tick_data, period):

    last_df = tick_data.groupby('bar_id').agg(
        Close=('Last', 'last'),
    ).reset_index()

    last_df['SMA_sum'] = (last_df['Close'].rolling(window=(period - 1), min_periods=1).sum().shift())

    last_df.set_index('bar_id', inplace=True) # removing this line squews the results
    tick_data['last_sma_sum'] = tick_data['bar_id'].map(last_df['SMA_sum']).astype('float32')

    SMA = (tick_data['last_sma_sum'] + tick_data['Last']) / period

    tick_data.drop(columns= ['last_sma_sum'], inplace=True, errors='ignore')

    return SMA

# -------------------------------------------------------------------   
# ----------------------- Main Functions ----------------------------
# -------------------------------------------------------------------

def find_stochastic(tick_data, params):
    
    k_length = params['k_length']

    # Downcast 'Volume' and 'Last' to float32
    tick_data['Volume'] = tick_data['Volume']
    tick_data['Last'] = tick_data['Last']

    # Ensure 'datetime' is datetime64[ns]
    tick_data['datetime'] = pd.to_datetime(tick_data['datetime'], errors='coerce')

    # Start Stochastic Oscillator calculation
    Stoch_time = time.time()

    # Convert 'bar_start' to boolean
    tick_data['bar_start'] = tick_data['Volume'].isna().astype(bool)

    # 'bar_id' as unsigned integer
    tick_data['bar_id'] = tick_data['bar_start'].cumsum().astype('float32')

    # Remove rows where 'Volume' is NaN
    tick_data.dropna(subset=['Volume'], inplace=True)

    # 'bar_count' as unsigned integer
    tick_data['bar_count'] = tick_data.groupby('bar_id').cumcount() + 1
    tick_data['bar_count'] = tick_data['bar_count'].astype('float32')
    
    tick_data['cumulative_high'] = tick_data.groupby('bar_id')['Last'].cummax().astype('float32').round(2)
    tick_data['cumulative_low'] = tick_data.groupby('bar_id')['Last'].cummin().astype('float32').round(2)

    tick_data['high_low_diff'] = (tick_data['cumulative_high'] - tick_data['cumulative_low']).astype('float32').round(2)

    ohlc_bars = tick_data.groupby('bar_id').agg(
        High=('Last', 'max'),
        Low=('Last', 'min'),
    ).reset_index().astype({'bar_id': 'uint32'})

    ohlc_bars['highest_high'] = (ohlc_bars['High'].rolling(window=(k_length - 1), min_periods=1).max().shift()).astype('float32').round(2)
    ohlc_bars['lowest_low'] = (ohlc_bars['Low'].rolling(window=(k_length - 1), min_periods=1).min().shift()).astype('float32').round(2)

    ohlc_bars.set_index('bar_id', inplace=True)
    tick_data['prev_highest_high'] = tick_data['bar_id'].map(ohlc_bars['highest_high']).astype('float32').round(2)
    tick_data['prev_lowest_low'] = tick_data['bar_id'].map(ohlc_bars['lowest_low']).astype('float32').round(2)

    tick_data['highest_high'] = np.maximum(tick_data['prev_highest_high'], tick_data['High']).astype('float32').round(2)
    tick_data['lowest_low'] = np.minimum(tick_data['prev_lowest_low'], tick_data['Low']).astype('float32').round(2)

    price_range = tick_data['highest_high'] - tick_data['lowest_low']
    tick_data['Fast_K'] = np.where(
        price_range != 0,
        (tick_data['Last'] - tick_data['lowest_low']) / price_range * 100,
        50  # Default value if no price movement
    )

    last_tick_fast_k = tick_data.loc[tick_data.groupby('bar_id')['bar_count'].idxmax(), ['bar_id', 'Fast_K']]
    last_tick_fast_k = last_tick_fast_k.rename(columns={'Fast_K': 'Final_Fast_K'})
    last_tick_fast_k.set_index('bar_id', inplace=True)

    ohlc_bars = ohlc_bars.join(last_tick_fast_k, how='left')

    ohlc_bars['Fast_K_previous_1'] = ohlc_bars['Final_Fast_K'].shift(1).astype('float32').round(2)
    ohlc_bars['Fast_K_previous_2'] = ohlc_bars['Final_Fast_K'].shift(2).astype('float32').round(2)

    tick_data['Fast_K_previous_1'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_1']).astype('float32').round(2)
    tick_data['Fast_K_previous_2'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_2']).astype('float32').round(2)

    # slow k with 2 decimals
    tick_data['Slow_K'] = (tick_data[['Fast_K', 'Fast_K_previous_1', 'Fast_K_previous_2']].mean(axis=1)).astype('float32').round(2)

    lookback = params['big_k']
    tick_data['Big_k'] = stochastic(lookback, tick_data)

    Stoch_time_finished = time.time() - Stoch_time
    # print(f'Stochastic Oscillator calculation finished. Elapsed time: {Stoch_time_finished:.2f} seconds')

    # Drop unnecessary columns to free up memory
    cols_to_drop_after_stoch = [
        'bar_start', 'cumulative_high', 'cumulative_low',
        'highest_high', 'lowest_low', 'Fast_K_previous_1', 'Fast_K_previous_2',
        'Fast_K', 'Final_Fast_K', 'prev_highest_high', 'prev_lowest_low'
    ]
    tick_data.drop(columns=cols_to_drop_after_stoch, inplace=True, errors='ignore')
    del ohlc_bars, last_tick_fast_k, cols_to_drop_after_stoch, price_range, Stoch_time, Stoch_time_finished, k_length

    # fill skowk and bigk null values with 0
    tick_data['Slow_K'] = tick_data['Slow_K'].fillna(0)
    tick_data['Big_k'] = tick_data['Big_k'].fillna(0)

    return tick_data

def find_AverageRate(tick_data, params):
    # Compute Average rate of the last 5 and 15 price changes
    # C:\Projects\StrategyBuilder\modules\Strat__History\AvgRate_only.py
    changes_time = time.time()
    
    # Compute Average Rate including periods of unchanged price
    num_changes_5 = params['num_changes_5']

    tick_data['price_changed'] = tick_data['Last'].diff() != 0

    tick_data['price_change_count'] = tick_data['price_changed'].cumsum()

    price_change_times = tick_data.loc[tick_data['price_changed'], ['price_change_count', 'datetime']].astype({'price_change_count': 'uint32'})
    price_change_times = price_change_times.set_index('price_change_count')['datetime']

    # For AvgRate5
    N = num_changes_5
    tick_data['price_change_count_N'] = (tick_data['price_change_count'] - N).astype('float32').clip(lower=0)
    tick_data['datetime_N'] = tick_data['price_change_count_N'].map(price_change_times)

    del price_change_times

    tick_data['time_since_Nth_price_change'] = ((tick_data['datetime'] - tick_data['datetime_N']).dt.total_seconds() * 1000).astype('float32') # in milliseconds

    # Handle NaT values
    tick_data['time_since_Nth_price_change'] = tick_data['time_since_Nth_price_change'].ffill().astype('float32')

    tick_data['AvgRate5'] = (tick_data['time_since_Nth_price_change'] / N).astype('float32')
    tick_data['AvgRate5'] = tick_data['AvgRate5'].ffill()
    tick_data['AvgRate5'] = tick_data['AvgRate5'].astype('float32')
    tick_data['AvgRate5'] = tick_data['AvgRate5'].fillna(0).astype('float32')
    
    tick_data['AvgRate5_at_price_change'] = np.nan

    price_change_indices = tick_data.index[tick_data['price_changed']]
    
    # Assign 'AvgRate5' values at price change indices
    tick_data.loc[price_change_indices, 'AvgRate5_at_price_change'] = tick_data.loc[price_change_indices, 'AvgRate5']
    
    # Shift the 'AvgRate5_at_price_change' column
    tick_data['AvgRate5_at_previous_price_change'] = tick_data['AvgRate5_at_price_change'].shift(1)
    tick_data['AvgRate5_at_previous_price_change'] = tick_data['AvgRate5_at_previous_price_change'].astype('float32')
    tick_data['AvgRate5_at_previous_price_change'] = tick_data['AvgRate5_at_previous_price_change'].ffill().astype('float32')

    # Calculate AvgRate5_RoC from the previous price change
    tick_data['AvgRate5_RoC'] = ((
        (tick_data['AvgRate5'] - tick_data['AvgRate5_at_previous_price_change']) /
        tick_data['AvgRate5_at_previous_price_change'].replace(0, np.nan)
    ) * 100).astype('float32')

    # Handle division by zero or NaN values
    tick_data['AvgRate5_RoC'] = tick_data['AvgRate5_RoC'].fillna(0).astype('float32')

    # **Calculate 'is_slowing_down'**
    slowing_threshold = params.get('AvgRate5_RoC_Slowing_Threshold', params['RoC_Threshold'])  # Adjust threshold as needed
    tick_data['is_slowing_down'] = tick_data['AvgRate5_RoC'] > slowing_threshold

    # Optional: Remove temporary columns
    tick_data.drop([
        'AvgRate5_at_price_change', 'AvgRate5_at_previous_price_change', 'price_changed', 
        'price_change_count', 'price_change_count_N', 'datetime_N', 'time_since_Nth_price_change'
    ], axis=1, inplace=True)

    changes_time_finished = time.time() - changes_time
    # print(f'Average rate calculation finished. Elapsed time: {changes_time_finished:.2f} seconds')

    return tick_data

def find_ComplexValue(tick_data, params):
    # print(f'Average rate calculation finished. Elapsed time: {changes_time_finished:.2f} seconds')
    Complex_time = time.time()
    candles = params['candles']
    high_average = params['high_average']
    high_value = params['high_value']
    low_average = params['low_average']
    low_value = params['low_value']
    threshold = params['threshold']
    min_value = params['min_value']
    max_value = params['max_value']

    max_bar_count_indices = tick_data.groupby('bar_id')['bar_count'].idxmax()
    max_bar_diff = tick_data.loc[max_bar_count_indices, ['bar_id', 'high_low_diff']].reset_index(drop=True)
    max_bar_diff['average_size'] = max_bar_diff['high_low_diff'].rolling(window=candles, min_periods=candles).mean().shift(1) * 4

    tick_data = tick_data.merge(max_bar_diff[['bar_id', 'average_size']], on='bar_id', how='left')
    tick_data['average_size'] = tick_data.groupby('bar_id')['average_size'].ffill()

    del max_bar_diff, max_bar_count_indices

    tick_data['ComplexValue'] = tick_data['average_size'].apply(
        lambda avg_size: calculate_complex_value(avg_size, high_average, high_value, low_average, low_value, threshold, min_value, max_value)
    )

    tick_data.drop(['price_changed', 'elapsed','average_size', 'high_low_diff'], axis=1, inplace=True, errors='ignore')

    Complex_time_finished = time.time() - Complex_time
    # print(f'Complex value calculation finished. Elapsed time: {Complex_time_finished:.2f} seconds')

    return tick_data

def find_donchian(tick_data, params):
    period = params['donchian_period']  # Default period is 4

    # Ensure 'datetime' is datetime64[ns]
    tick_data['datetime'] = pd.to_datetime(tick_data['datetime'], errors='coerce')

    # Calculate cumulative high and low within each bar
    tick_data['cumulative_high'] = tick_data.groupby('bar_id')['Last'].cummax()
    tick_data['cumulative_low'] = tick_data.groupby('bar_id')['Last'].cummin()

    # Get High and Low per bar
    ohlc_bars = tick_data.groupby('bar_id').agg(
        High=('Last', 'max'),
        Low=('Last', 'min')
    ).reset_index()

    # Get Highs and Lows of previous periods
    for i in range(1, period):
        ohlc_bars[f'High_shift_{i}'] = ohlc_bars['High'].shift(i)
        ohlc_bars[f'Low_shift_{i}'] = ohlc_bars['Low'].shift(i)

    # Merge shifted Highs and Lows into tick_data
    shifted_cols = ['bar_id'] + [f'High_shift_{i}' for i in range(1, period)] + [f'Low_shift_{i}' for i in range(1, period)]
    tick_data = tick_data.merge(ohlc_bars[shifted_cols], on='bar_id', how='left')

    # Compute Donchian upper and lower bands
    high_columns = [f'High_shift_{i}' for i in range(1, period)] + ['cumulative_high']
    low_columns = [f'Low_shift_{i}' for i in range(1, period)] + ['cumulative_low']
    tick_data['donchian_upper'] = tick_data[high_columns].max(axis=1)
    tick_data['donchian_lower'] = tick_data[low_columns].min(axis=1)

    # Clean up
    cols_to_drop = [col for col in tick_data.columns if 'shift' in col or 'cumulative_' in col]
    tick_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return tick_data


def find_VHF(tick_data, params):
    # Compute Volatility Hypertrend Factor (VHF)
    # "VHF" determines whether prices are in a trending phase or a congestion phase.

    VHF_period = params['VHF_period']

    ohlc_bars = tick_data.groupby('bar_id').agg(
        Close=('Last', 'last'),
    ).reset_index().astype({'bar_id': 'uint32'})

    ohlc_bars['max_period'] = (ohlc_bars['Close'].rolling(window=(VHF_period - 1), min_periods=1).max().shift()).astype('float32').round(2)
    ohlc_bars['min_period'] = (ohlc_bars['Close'].rolling(window=(VHF_period - 1), min_periods=1).min().shift()).astype('float32').round(2)
    
    ohlc_bars.set_index('bar_id', inplace=True)
    tick_data['prev_max_period'] = tick_data['bar_id'].map(ohlc_bars['max_period']).astype('float32').round(2)
    tick_data['prev_min_period'] = tick_data['bar_id'].map(ohlc_bars['min_period']).astype('float32').round(2)

    tick_data['max_period'] = np.maximum(tick_data['prev_max_period'], tick_data['Last']).astype('float32').round(2)
    tick_data['min_period'] = np.minimum(tick_data['prev_min_period'], tick_data['Last']).astype('float32').round(2)  
 
    enumerator = abs(tick_data['max_period'] - tick_data['min_period']) 
    tick_data['enumerator'] = enumerator      

    # difference in the provious OHLC bars closing price
    ohlc_bars['close_diff'] = ohlc_bars['Close'].diff().abs()
    # sum of the past VHF_period differences
    ohlc_bars['sum_period'] = (ohlc_bars['close_diff'].rolling(window=(VHF_period -1), min_periods=1).sum().shift()).astype('float32').round(2)
    # previous close
    ohlc_bars['prev_close'] = ohlc_bars['Close'].shift(1)

    # merge the previous close, close difference and sum of the past VHF_period differences
    tick_data['prev_close'] = tick_data['bar_id'].map(ohlc_bars['prev_close']).astype('float32').round(2)
    tick_data['prev_sum_period'] = tick_data['bar_id'].map(ohlc_bars['sum_period']).astype('float32').round(2)


    denominator = (tick_data['prev_sum_period'] + abs(tick_data['Last'] - tick_data['prev_close']))
    tick_data['denominator'] = denominator

    tick_data['VHF'] = np.where(
        denominator != 0,
        enumerator / denominator,
        0  # Default value if denominator is zero
    )
    tick_data['VHF'] = tick_data['VHF'].fillna(0).astype('float32')

    # drop columns
    cols_to_drop = ['prev_max_period', 'prev_min_period', 'max_period', 'min_period', 'enumerator', 'prev_close', 'prev_sum_period', 'denominator']
    tick_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return tick_data


def find_SMAs(tick_data, params):

    sma1_period = params['sma1_period']
    sma2_period = params['sma2_period']

    tick_data['sma_1'] = SMA(tick_data, sma1_period).fillna(0)
    tick_data['sma_2'] = SMA(tick_data, sma2_period).fillna(0)


    return tick_data

def add_seconds_since_midnight(tick_data):
    tick_data['sec_sm'] = (
        tick_data['datetime'].dt.hour * 3600
        + tick_data['datetime'].dt.minute * 60
        + tick_data['datetime'].dt.second
        + tick_data['datetime'].dt.microsecond / 1e6
    ).round(3).astype('float32')
    return tick_data

# -------------------------------------------------------------------
# ------------------------ Main Process -----------------------------
# -------------------------------------------------------------------

def process_tick_data(tick_data, params):

    tick_data = find_stochastic(tick_data, params) 
    # tick_data = find_AverageRate(tick_data, params) 
    # tick_data = find_ComplexValue(tick_data, params)
    # tick_data = find_donchian(tick_data, params)
    # tick_data = find_VHF(tick_data, params)
    # tick_data = find_SMAs(tick_data, params)
    # Call it after loading or preparing tick_data
    tick_data = add_seconds_since_midnight(tick_data)

    # apply valuemarker coulmn
    tick_data['valuemarker'] = np.nan

    return tick_data


