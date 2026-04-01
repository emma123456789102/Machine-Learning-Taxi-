import dataset
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt

#################################################################################################
#
# Collection of regression scores
#
# 1. Mean Abs Err: 
# 2. Mean Sq Err:
# 3. R Sq: percentage of variance explained
# 4. Residuals plot
# 5. Actual vs. Predicted plot
def analysis(y_true, y_pred):
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Absolute Error: \t{mae:.4f}")

    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    print(f"Mean Squared Error: \t{mse:.4f}")

    # R squared
    r2 = r2_score(y_true, y_pred)
    print(f"R squared: \t\t{r2:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals
    residuals = y_true - y_pred
    ax[0].scatter(y_pred, residuals, alpha=.4)
    ax[0].axhline(0, color='black', linestyle='--')
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Residuals")
    ax[0].set_title("Residuals")

    # Actual vs predicted
    ax[1].scatter(y_true, y_pred, alpha=.4)
    ax[1].set_xlabel("Actual")
    ax[1].set_ylabel("Predicted")
    ax[1].set_title("Actual vs. Predicted")


#################################################################################################
#
# Return `total_ride_count` as y variable, and encoded training set
def preprocess():
    df = dataset.read_agg(month_start=1, month_end=12)
    
    # Recast object types -> category
    # Not sure why this changes from typecasting in `dataset.clean()`
    df = df.astype({
        'vendor': 'category',
        # 'ratecode': 'category',
        'pickup_zone': 'category',
        'dropoff_zone': 'category',
        'route': 'category',
    })

    # Keep only categorical data
    df = df.select_dtypes(exclude='float64')

    # Onehot encode the following:
    #
    #  4   pickup_service_zone    category
    #  5   pickup_zone            category  
    #  6   dropoff_service_zone   category
    #  7   dropoff_zone           category  
    #  8   route                  category  
    #  9   service_route          category
    #  10  vendor                 category  
    #  11  ratecode               category  
    #  12  payment_type           category
    #
    # These columns only have a few categories
    #
    # => onehot encoding is okay
    columns_to_onehot_encode = [
        'pickup_service_zone',
        'dropoff_service_zone',
        'service_route',
        'vendor',
        # 'ratecode',
        # 'payment_type',
    ]

    # These columns have LOADS of categories
    #
    # => ordinal encoding is needed             <---- not doing this gives 500GB encoded dataset
    #``
    # (tried TargetEncoder, but still ended up at 22GB - sticking to ordinal)
    columns_to_ordinal_encode = [
        'pickup_zone',
        'dropoff_zone',
        'route'
    ]

    # Split `y` BEFORE pipeline
    y = df['total_ride_count']
    X = df.drop(columns=['total_ride_count'])

    # Pipeline
    #
    # Transfroms the categorical columns to onehot-encoded types
    pipeline = ColumnTransformer(
        transformers=[
            #                               +--- `sparse_output=True` stops the encoder duplicating data
            #                               |
            ("onehot", OneHotEncoder(sparse_output=True), columns_to_onehot_encode), 
            ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), columns_to_ordinal_encode),
        ],
        remainder='passthrough'
    )
    # pipeline.set_output(transform='pandas') # DEPRECATED: returning pandas cost too much memory
    X_encoded = pipeline.fit_transform(X)
    
    return y, X_encoded