import pandas as pd

# Return only one month from `data/` dircectory.
#
# <https://studres.cs.st-andrews.ac.uk/ID5059/Coursework/P2/data/>
# <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
def read_single_month(month=1):
    path = f"../../data/nyc_taxi_2024-{month:02d}.csv"
    df = pd.read_csv(
        path, 
        parse_dates = ["tpep_pickup_datetime", "tpep_dropoff_datetime"],
        dtype = {"store_and_fwd_flag": str}
    )
    return df

##############################################################################
#
# Read all 12 csvs into 1 dataframe
#
# `nyc_taxi_2024-*.csv` & `taxi_zone_lookup.csv` expected in the `data/` directory
#
# <https://studres.cs.st-andrews.ac.uk/ID5059/Coursework/P2/data/>
# <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
def raw(month_start=1, month_end=2):
    df = pd.DataFrame() # Empty, base df

    # `month_end+1` because the loop stops 1 before end
    for month in range(month_start, month_end+1):
        tmp = read_single_month(month=month)
        df = pd.concat([df, tmp], ignore_index=True) 

    return df

##############################################################################
#
# Transformed version of the dataset, but still **unaggregated**.
#
# 1. cleans dataset for: invalid fares, invalid distances, where pickup is after dropoff
#    and missing pickup items
#
# 2. transforms:
#   a. timeperiod aggregates (what hour, day, week, etc.)
#   b. change `store_and_fwd_flag` from {'Y', 'N'} -> True/False
#   c. real `ratecode` from ID
#   d. real `payment_type` from ID
#   e. real `vendor` from ID
#   f. real `taxi zone` from ID (both pickup zone & dropoff zone)
#   g. real `service zone` from ID (more abstract version of `taxi zone`)
#   h. include `route` taken, i.e. string of `pickup zone` + `dropoff zone`
#   i. same as (h.), but for service zones, as well
#
# 3. drop ID columns
#
# 4. typecast all `object` types -> `category` (expect `store_and_fwd_flag`, which
#    is a boolean)
def clean(month_start=1, month_end=2):
    df = raw(month_start=month_start, month_end=month_end)

    # ----------------------------------------------------------------------------#

    ### CLEANING

    # Removing rows with invalid fares (negative or zero fare amounts don't make sense)
    df = df[df["fare_amount"] > 0]

    # Remove rows with invalid trip distances
    df = df[df["trip_distance"] > 0]

    # Remove rows where pickup is after dropoff (data error)
    df = df[df["tpep_pickup_datetime"] < df["tpep_dropoff_datetime"]]

    # Remove any rows with missing pickup times
    df = df.dropna(subset=["tpep_pickup_datetime"])

    # ----------------------------------------------------------------------------#

    ### TRANSFORMATIONS

    # Time aggregates
    df['pickup_hr'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
    df['pickup_week'] = df['tpep_pickup_datetime'].dt.isocalendar().week
    df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
    df['pickup_year'] = df['tpep_pickup_datetime'].dt.year

    df['dropoff_hr'] = df['tpep_dropoff_datetime'].dt.hour
    df['dropoff_day'] = df['tpep_dropoff_datetime'].dt.day
    df['dropoff_week'] = df['tpep_dropoff_datetime'].dt.isocalendar().week
    df['dropoff_month'] = df['tpep_dropoff_datetime'].dt.month
    df['dropoff_year'] = df['tpep_dropoff_datetime'].dt.year

    # store_and_fwd_flag
    #
    # This flag indicates whether the trip record was held in vehicle memory before
    # sending to the vendor, aka “store and forward,” because the vehicle did not
    # have a connection to the server.
    # Y = store and forward trip
    # N = not a store and forward trip
    #
    # Change flag from 'Y' or 'N' -> True or False
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y': True, 'N': False})

    # RatecodeID
    #
    # The final rate code in effect at the end of the trip.
    # 1 = Standard rate
    # 2 = JFK
    # 3 = Newark
    # 4 = Nassau or Westchester
    # 5 = Negotiated fare
    # 6 = Group ride
    # 99 = Null/unknown
    ratecode_mapping = {
        1: "Standard Rate",
        2: "JFK",
        3: "Newark",
        4: "Nassua or Westchester",
        5: "Negotiated fare",
        6: "Group Ride",
        99: "Null/unknown",
    }
    df['ratecode'] = df['RatecodeID'].map(ratecode_mapping)

    # payment_type
    #
    # A numeric code signifying how the passenger paid for the trip.
    # 0 = Flex Fare trip
    # 1 = Credit card
    # 2 = Cash
    # 3 = No charge
    # 4 = Dispute
    # 5 = Unknown
    # 6 = Voided trip
    payment_type_mapping = {
        0: "Flex Fare trip",
        1: "Credit card",
        2: "Cash",
        3: "No charge",
        4: "Dispute",
        5: "Unknown",
        6: "Voided trip",
    }
    df['payment_type'] = df['payment_type'].map(payment_type_mapping)

    # VendorID
    #
    # A code indicating the TPEP provider that provided the record.
    # 1 = Creative Mobile Technologies, LLC
    # 2 = Curb Mobility, LLC
    # 6 = Myle Technologies Inc
    # 7 = Helix
    vendor_mapping = {
        1: "Creative Mobile Technologies, LLC",
        2: "Curb Mobility, LLC",
        6: "Myle Technologies Inc",
        7: "Helix",
    }
    df['vendor'] = df['VendorID'].map(vendor_mapping)    

    # Taxi zones
    #
    # .csv file needed to lookup codes
    zones = pd.read_csv("../../data/taxi_zone_lookup.csv")
    manhattan_ids = zones[zones["Borough"] == "Manhattan"]["LocationID"].tolist()

    # Taxi zones: Pickup zone & pickup service zone
    df = df[df["PULocationID"].isin(manhattan_ids)]
    df = df.merge(
        zones[['LocationID', 'Zone', 'service_zone']].rename(columns={
            'Zone': 'pickup_zone',
            'service_zone': 'pickup_service_zone',
        }),
        left_on='PULocationID', 
        right_on='LocationID',
        how='left'
    )
    df = df.drop(columns=['LocationID'])

    # Taxi zones: Dropoff zone & dropoff service zone
    df = df.merge(  
        zones[['LocationID', 'Zone', 'service_zone']].rename(columns={
            'Zone': 'dropoff_zone',
            'service_zone': 'dropoff_service_zone',
        }),
        left_on='DOLocationID', 
        right_on='LocationID',
        how='left'
    )
    df = df.drop(columns=['LocationID'])

    # Taxi zones: route taken
    df['route'] = df['pickup_zone'].astype(str) + " to " + df['dropoff_zone'].astype(str)
    df['service_route'] = df['pickup_service_zone'].astype(str) + " to " + df['dropoff_service_zone'].astype(str)

    # ----------------------------------------------------------------------------#

    ### DROP COLUMNS
    columns_to_drop = [
        'RatecodeID',
        'VendorID',
        'PULocationID',
        'DOLocationID',
    ]
    df = df.drop(columns=columns_to_drop)


    ### TYPE CAST
    df = df.astype({
        'store_and_fwd_flag': 'bool',
        'payment_type': 'category',
        'ratecode': 'category',
        'vendor': 'category',
        'pickup_zone': 'category',
        'pickup_service_zone': 'category',
        'dropoff_zone': 'category',
        'dropoff_service_zone': 'category',
        'route': 'category',
        'service_route': 'category',
    })

    return df