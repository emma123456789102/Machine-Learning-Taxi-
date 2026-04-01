import pandas as pd
import requests
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

##############################################################################
#
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
def clean_single_month(month=1):
    # df = raw(month_start=month_start, month_end=month_end)
    df = read_single_month(month=month)

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
    #
    # Time aggregates
    df['pickup_date'] = pd.to_datetime(df['tpep_pickup_datetime'].dt.date)
    df['pickup_hr'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.day
    df['pickup_dow'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_week'] = df['tpep_pickup_datetime'].dt.isocalendar().week
    df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
    df['pickup_year'] = df['tpep_pickup_datetime'].dt.year

    df['dropoff_hr'] = df['tpep_dropoff_datetime'].dt.hour
    df['dropoff_day'] = df['tpep_dropoff_datetime'].dt.day
    df['dropoff_dow'] = df['tpep_dropoff_datetime'].dt.weekday
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


##############################################################################
#
# Documentation:
#     <https://mesonet.agron.iastate.edu/nws/cf6table.php?station=KNYC&opt=bystation&year=2024>
#
# The API specifically calls:
#     <https://mesonet.agron.iastate.edu/json/cf6.py?station=KNYC&year=2024>
#
def read_weather_data():
    # Using new API
    response = requests.get("https://mesonet.agron.iastate.edu/json/cf6.py?station=KNYC&year=2024").json()
    wthr = pd.DataFrame(response['results'])
    wthr['date'] = pd.to_datetime(wthr['valid'])
    wthr = wthr.drop(columns=['name', 'station', 'valid', 'state', 'wfo', 'link', 'product', 'minutes_sunshine', 'possible_sunshine', 'hdd', 'cdd', 'gust_drct', 'avg_drct', 'snowd_12z', 'avg_smph'])

    # CF6 Code	Abbrev	Meaning
    # 1	FG	Fog or Mist
    # 2	DNSEFG	Fog or Vis 0.25 mile or less
    # 3	TS	Thunder
    # 4	IP	Ice pellets
    # 5	GR	Hail
    # 6	FZRA	Freezing Rain or Drizzle
    # 7	DSTSTM	Duststorm or Sandstorm vis 0.25 mile or less
    # 8	HZ	Smoke or Haze
    # 9	BLSN	Blowing Snow
    # X	TOR	Tornado
    # M	M	Missing Data
    wthr['fog']           = wthr['wxcodes'].str.contains('1', na=False).astype(bool)
    wthr['low_vis']       = wthr['wxcodes'].str.contains('2', na=False).astype(bool)
    wthr['thunder']       = wthr['wxcodes'].str.contains('3', na=False).astype(bool)
    wthr['ice']           = wthr['wxcodes'].str.contains('4', na=False).astype(bool)
    wthr['hail']          = wthr['wxcodes'].str.contains('5', na=False).astype(bool)
    wthr['freezing_rain'] = wthr['wxcodes'].str.contains('6', na=False).astype(bool)
    wthr['duststorm']     = wthr['wxcodes'].str.contains('7', na=False).astype(bool)
    wthr['haze']          = wthr['wxcodes'].str.contains('8', na=False).astype(bool)
    wthr['blowing_snow']  = wthr['wxcodes'].str.contains('9', na=False).astype(bool)
    wthr['tornado']       = wthr['wxcodes'].str.contains('X', na=False).astype(bool)

    # A lot of numerical values can include 'T' as an entry
    #
    # This refers to 'trace amounts', i.e. miniscule levels of precipitation; we will just replace with 0
    wthr.replace('T', 0, inplace=True)
    wthr.replace('M', np.nan, inplace=True)
    wthr = wthr.infer_objects(copy=False)
    wthr = wthr.astype({
        'snow': 'float64',
        'precip': 'float64',
    })

    wthr = wthr.drop(columns=[
        'wxcodes', # redundant
        'ice', # empty
        'tornado', # empty
        'blowing_snow', # empty
        'duststorm', # empty
        'avg_temp', # using 'high'
        'dep_temp', # using 'high'
        'low', # using 'high'
        'gust_smph', # using 'max_smph'
    ])

    wthr = wthr.rename(columns={
        'high': 'temp_high',
        'max_smph': 'max_wind_speed',
        'cloud_ss': 'cloud_coverage',
    })

    return wthr

##############################################################################
#
# Source: <https://www.officeholidays.com/countries/usa/new-york/2024>
#
# | Date | Holiday Name | Holiday Type |
# |------|--------------|--------------|
# | 01/01/2024 | New Year's Day | National |
# | 15/01/2024 | Martin Luther King Jr. Day | National |
# | 12/02/2024 | Lincoln's Birthday | Government |
# | 19/02/2024 | Washington's Birthday | Regional |
# | 31/03/2024 | Easter Sunday | Not a holiday |
# | 12/05/2024 | Mother's Day | Not a holiday |
# | 27/05/2024 | Memorial Day | National |
# | 16/06/2024 | Father's Day | Not a holiday |
# | 19/06/2024 | Juneteenth | Regional |
# | 04/07/2024 | Independence Day | National |
# | 02/09/2024 | Labor Day | National |
# | 14/10/2024 | Columbus Day | Regional |
# | 05/11/2024 | Election Day | Government |
# | 11/11/2024 | Veterans Day | Regional |
# | 28/11/2024 | Thanksgiving | National |
# | 25/12/2024 | Christmas Day | National |
def calendar():
    data = {
        "date": [
            "01/01/2024", "15/01/2024", "12/02/2024", "19/02/2024",
            "31/03/2024", "12/05/2024", "27/05/2024", "16/06/2024",
            "19/06/2024", "04/07/2024", "02/09/2024", "14/10/2024",
            "05/11/2024", "11/11/2024", "28/11/2024", "25/12/2024"
        ],
        "holiday": [
            "New Year's Day", "Martin Luther King Jr. Day", "Lincoln's Birthday",
            "Washington's Birthday", "Easter Sunday", "Mother's Day",
            "Memorial Day", "Father's Day", "Juneteenth", "Independence Day",
            "Labor Day", "Columbus Day", "Election Day", "Veterans Day",
            "Thanksgiving", "Christmas Day"
        ],
        "holiday_type": [
            "National", "National", "Government", "Regional", "Not a holiday",
            "Not a holiday", "National", "Not a holiday", "Regional", "National",
            "National", "Regional", "Government", "Regional", "National", "National"
        ]
    }

    calendar = pd.DataFrame(data)
    calendar["date"] = pd.to_datetime(calendar["date"], format="%d/%m/%Y")
    return calendar

##############################################################################
#
# For many months, clean, aggregated, and append monthly data.
groupings = [
    'pickup_year',
    'pickup_month',
    'pickup_week', 
    'pickup_day',
    'pickup_date',
    'pickup_dow',
    # 'pickup_hr', 
    'pickup_service_zone', 
    'pickup_zone',
    'dropoff_service_zone', 
    'dropoff_zone',
    'route',
    'service_route',
    'vendor', 
    # 'ratecode', 
    # 'payment_type',
    # 'store_and_fwd_flag'
]
def read_agg(month_start=1, month_end=2, groupings=groupings):
    base_df = pd.DataFrame()

    for month in range(month_start, month_end+1):
        tmp_df = clean_single_month(month)
        agg_df = tmp_df.groupby(groupings, as_index=False, observed=True).agg(
            total_ride_count        = ('tpep_pickup_datetime', 'count'), # count number of rides
            total_passenger_count   = ('passenger_count', 'sum'),
            avg_passenger_count     = ('passenger_count', 'mean'),
            total_trip_distance     = ('trip_distance', 'sum'),
            avg_trip_distance       = ('trip_distance', 'mean'),
            total_fare_amount       = ('fare_amount', 'sum'),
            avg_fare_amount         = ('fare_amount', 'mean'),
            total_extra             = ('extra', 'sum'),
            avg_extra               = ('extra', 'mean'),
            total_mta_tax           = ('mta_tax', 'sum'),
            avg_mta_tax             = ('mta_tax', 'mean'),
            total_tip_amount        = ('tip_amount', 'sum'),
            avg_tip_amount          = ('tip_amount', 'mean'),
            total_tolls_amount      = ('tolls_amount', 'sum'),
            avg_tolls_amount        = ('tolls_amount', 'mean'),
            total_impr_surcharge    = ('improvement_surcharge', 'sum'),
            avg_impr_surcharge      = ('improvement_surcharge', 'mean'),
            total_revenue           = ('total_amount', 'sum'),
            avg_revenue             = ('total_amount', 'mean'),
            total_airport_fee       = ('Airport_fee', 'sum'),
            avg_airport_fee         = ('Airport_fee', 'mean'),
        )
        base_df = pd.concat([base_df, agg_df], ignore_index=True)

    weather = read_weather_data()
    base_df = pd.merge(base_df, weather, left_on='pickup_date', right_on='date', how='left')
    base_df = base_df.drop(columns=['date'])

    cal = calendar()
    base_df = pd.merge(base_df, cal, left_on='pickup_date', right_on='date', how='left')
    base_df['holiday'] = base_df['holiday'].fillna("None")
    base_df = base_df.drop(columns=['date', 'holiday_type'])

    return base_df