import pandas as pd
import numpy as np


def time_filter(dframe, TIME_MAX, UNIT):
    '''
    This function is used to filter events according to their duration.

    Parameters:
    dframe: type(DataFrame) -> Tracking DataFrame
    TIME_MAX: type(Int) -> Maximum time value
    UNIT: type(Str) -> Time unit ('min' = minute, 'h' = hour)

    Output:
    return: type(DataFrame) -> Time filtered track events 
    '''
    uids_ = []
    dframe.groupby('uid').apply(lambda x: uids_.extend(x.uid.values) if
                                pd.to_datetime(
                                    x.iloc[-1].timestamp) - pd.to_datetime(x.iloc[0].timestamp)
                                >= pd.Timedelta(TIME_MAX, unit=UNIT) else None)

    uids_ = np.unique(uids_)[np.unique(uids_) != -1].tolist()
    return dframe.query('uid == @uids_')
