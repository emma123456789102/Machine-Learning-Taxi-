import pandas as pd

# `nyc_taxi_2024-*.csv` expected in the `data/` directory
def raw(month_start=1, month_end=12):
    # Read all 12 csvs into 1 dataframe
    df = pd.DataFrame() # Empty, base df

    # `month_end+1` because the loop stops 1 before end
    for month in range(month_start, month_end+1):
        path = f"../../data/nyc_taxi_2024_{month:02d}.csv"
        tmp = pd.read_csv(
            path, 
            parse_dates = ["tpep_pickup_datetime", "tpep_dropoff_datetime"],
            dtype = {"store_and_fwd_flag": str}
        )
        tmp["month"] = month

        # add each dataframe in to the base
        df = pd.concat([df, tmp], ignore_index=True) 

    # Column transformations


    return df