import ta

'''This function uses ta library to add features (technical analysis) to OHLCV data
Currently every possible ta feature is added.

Accepts: Dataframe
Returns: Dataframe
'''


def add_ta(ticker_data):
    df = ta.add_all_ta_features(ticker_data, open=f"open", high=f"high", low=f"low", close=f"close", volume=f"volume",
                                fillna=True, vectorized=False)  # Add all the ta!

    return df
