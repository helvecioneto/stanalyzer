import pandas as pd


def life_cicle(dframe=None, sort=False):
    '''
    This function is used to calculate the duration of the tracked events contained in the tracking DataFrame.

    Parameters:
    dframe: type(DataFrame) -> DataFrame of tracking

    Output:
    return: type(DataFrame) -> Duration of tracked events
    '''
    try:
        if len(dframe) <= 0:
            print('There\'s no DataFrame or Empty!!')
            return None
        else:

            # Group by uid
            grouped_frame, life_time, uid_, start_, end_ = [], [], [], [], []
            for group in dframe.groupby(pd.Grouper(key="uid")):
                grouped_frame.append(group)

            # Calculate by initial time and final time
            for f in range(len(grouped_frame)):
                life_time.append(len(grouped_frame[f][1]))
                uid_.append(grouped_frame[f][1].uid.values[0])
                start_.append(grouped_frame[f][1].timestamp.values[0]), end_.append(
                    grouped_frame[f][1].timestamp.values[-1])

            # Create cicle life dataframe
            cicle_life = pd.DataFrame(list(zip(uid_, life_time, start_, end_)),
                                      columns=['uid', 'times', 'begin', 'end'])
            # Cast to datetime
            cicle_life['begin'] = pd.to_datetime(cicle_life['begin'])
            cicle_life['end'] = pd.to_datetime(cicle_life['end'])

            # Calculate duration
            cicle_life['duration'] = pd.to_timedelta(
                cicle_life['end'] - cicle_life['begin'])

            if sort == True:
                return cicle_life.sort_values(by='times', ascending=False)
            else:
                return cicle_life
    except Exception as e:
        print('There\'s no DataFrame or Empty!!\n', e)
        return None
