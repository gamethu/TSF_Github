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

def extract_time_features(df, column_name):
    import pandas as pd
    
    # Đảm bảo cột time là kiểu datetime timezone-aware
    df[column_name] = pd.to_datetime(df[column_name])

    # Tạo cột ymd dạng 1/2/1990 ngay <~> 2 thang 1 namw 1990
    # df['ymd'] = df[column_name].dt.strftime('%-m/%-d/%Y')  # Unix-like

    # Nếu trên Windows thì dùng:
    df['ymd'] = df[column_name].dt.strftime('%#m/%#d/%Y')
    
    df['year']  = df[column_name].dt.year
    df['month'] = df[column_name].dt.month
    df['day']   = df[column_name].dt.day

    return df
