def utc_to_vietnam (df, column_name):
    import pandas as pd 
    from datetime import datetime
    from pytz     import timezone
    
    df[column_name] = pd.to_datetime(df[column_name]).dt.tz_localize ('UTC')
    df[column_name] = df[column_name].dt.tz_convert('Asia/Ho_Chi_Minh')
    return df