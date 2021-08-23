import pandas as pd
import numpy as np


def time_filter(dframe, TIME_MIN, TIME_MAX, UNIT):
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
    dframe.groupby('uid').apply(lambda x: uids_.extend(x.uid.values)
                                if pd.to_datetime(x.iloc[-1].timestamp) - pd.to_datetime(x.iloc[0].timestamp)
                                <= pd.Timedelta(TIME_MAX, unit=UNIT)
                                and
                                pd.to_datetime(
                                    x.iloc[-1].timestamp) - pd.to_datetime(x.iloc[0].timestamp)
                                >= pd.Timedelta(TIME_MIN, unit=UNIT) else None)

    uids_ = np.unique(uids_)[np.unique(uids_) != -1].tolist()
    return dframe.query('uid == @uids_')


def fam_type(df,stype='CONT'):

    '''
    This function is used to filter events according the status type.

    Parameters:
    dframe: type(DataFrame) -> Tracking DataFrame
    statys: string(CONT,SPLT or MERG)
    '''
    
    t0 = df.copy()
    t0 = t0.reset_index()
    
    all_CONT = []
    any_SPLS = []
    any_MERG = []
    for i,g in t0.groupby(pd.Grouper(key="uid")):
        ## IF ALL CONT
        if np.all(g.status.values == 'CONT'):
            all_CONT.append(g.status.index)
        stats=dict(zip(list(g.status.values),[list(g.status.values).count(i) for i in list(g.status.values)]))
        ## IF ONE OR MORE SPLTS
        if stats.get('CONT') != None and stats.get('CONT') > 0 and stats.get('SPLT') != None and stats.get('SPLT') > 0 and stats.get('MERG') == None:
            any_SPLS.append(g.status.index)
        ## IF ONE OR MORE MERG
        if stats.get('CONT') != None and stats.get('CONT') > 0 and stats.get('MERG') != None and stats.get('MERG') > 0 and stats.get('SPLT') == None:
            any_MERG.append(g.status.index)
            
    filterd_frame = pd.DataFrame()
    idx_list = []
    
    ## ONLY CONT
    if stype == 'CONT': 
        for idx in all_CONT:
            f1 = t0.loc[idx]
            filterd_frame = pd.concat([filterd_frame,f1])
        new_list = filterd_frame[['level_0','level_1']].values.tolist()
        mux = pd.MultiIndex.from_tuples(new_list)
        filterd_frame.index = mux
        filterd_frame = filterd_frame.drop(['level_0','level_1'], axis=1)
        return filterd_frame
    
    ## ONLY SPLT
    if stype == 'SPLT': 
        for idx in any_SPLS:
            f1 = t0.loc[idx]
            filterd_frame = pd.concat([filterd_frame,f1])
        new_list = filterd_frame[['level_0','level_1']].values.tolist()
        mux = pd.MultiIndex.from_tuples(new_list)
        filterd_frame.index = mux
        filterd_frame = filterd_frame.drop(['level_0','level_1'], axis=1)
        return filterd_frame
    
    ## ONLY MERG
    if stype == 'MERG': 
        for idx in any_MERG:
            f1 = t0.loc[idx]
            filterd_frame = pd.concat([filterd_frame,f1])
        new_list = filterd_frame[['level_0','level_1']].values.tolist()
        mux = pd.MultiIndex.from_tuples(new_list)
        filterd_frame.index = mux
        filterd_frame = filterd_frame.drop(['level_0','level_1'], axis=1)
        return filterd_frame