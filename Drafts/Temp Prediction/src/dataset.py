def grib_to_csv(path, des):
    import pandas as pd
    import cfgrib

    with cfgrib.open_dataset(path) as ds:
        df = ds.to_dataframe()
    df.to_csv(des)