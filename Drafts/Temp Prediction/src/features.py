def utc_to_vietnam(df, column_name):
    import pandas as pd 
    from pytz import timezone
    
    df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    if df[column_name].dt.tz is None:
        # Nếu chưa có timezone thì localize UTC trước
        df[column_name] = df[column_name].dt.tz_localize('UTC')
    else:
        # Nếu đã có timezone thì convert về Asia/Ho_Chi_Minh
        df[column_name] = df[column_name].dt.tz_convert('UTC')

    # Sau đó convert về Asia/Ho_Chi_Minh
    df[column_name] = df[column_name].dt.tz_convert('Asia/Ho_Chi_Minh')

    return df