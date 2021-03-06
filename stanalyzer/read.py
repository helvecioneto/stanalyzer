import pandas as pd

def read_file(path=None):
    '''
    This function is used for reading the tracking files (.pkl type) from Fortracc2.

    Parameters:
    path: type(String) -> Path to the track file with extension .pkl

    Output:
    return: type(DataFrame) -> The output of this function will be a DataFrame with the Fortrac trace variables
    '''
    try:
        return pd.read_pickle(path)
    except:
        try:
            return pd.read_pickle(path,compression='xz')
        except:
            print('File not found or path parameter is incorrect.') 
        print('File not found or path parameter is incorrect.') 
