import pandas as pd 

def load_data(path = 'data/employee_data.csv'):
    df = pd.read_csv(path)
    drop_col = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df.drop(columns  = drop_col, inplace = True, errors = 'ignore')
    df = pd.get_dummies(df, drop_first = True)
    return df 



