import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


airbnbData = pd.read_csv('dc_airbnb.csv')
airbnbData.drop(['city', 'zipcode', 'state'], axis = 1, inplace = True)
airbnbData.drop([1268, 2613, 3281], inplace = True)
airbnbData.fillna(0, inplace = True)

def percentage2int(x):
    if isinstance(x, str):
        x = x[:-1]
    return int(x)

def str2int(x):
    return int(x)

def room_type2int(x):
    if x == 'Entire home/apt':
        x = 1
    elif x == 'Private room':
        x = 2
    elif x == 'Shared room':
        x = 3
    return int(x)

def money2float(x):
    if isinstance(x, str):
        x = x[1:].replace(',', '')
    return float(x)

def str2float(x):
        if isinstance(x, str):
            x = x[1:]
        return float(x)

for featureOrLabel in ['host_response_rate', 'host_acceptance_rate', 'host_listings_count',
'accommodates', 'room_type', 'bedrooms', 'bathrooms', 'beds', 'price', 'cleaning_fee',
'security_deposit', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'latitude', 'longitude']:

    if featureOrLabel in ['host_response_rate', 'host_acceptance_rate']:
        airbnbData[featureOrLabel] = airbnbData[featureOrLabel].apply(percentage2int)

    elif featureOrLabel in ['host_listings_count', 'accommodates', 'bedrooms', 'bathrooms', 'beds',
    'minimum_nights', 'maximum_nights', 'number_of_reviews']:
        airbnbData[featureOrLabel] = airbnbData[featureOrLabel].apply(str2int)

    elif featureOrLabel in ['room_type']:
        airbnbData[featureOrLabel] = airbnbData[featureOrLabel].apply(room_type2int)

    elif featureOrLabel in ['price', 'cleaning_fee', 'security_deposit']:
        airbnbData[featureOrLabel] = airbnbData[featureOrLabel].apply(money2float)

    elif featureOrLabel in ['latitude', 'longitude']:
        airbnbData[featureOrLabel] = airbnbData[featureOrLabel].apply(str2float)

Pricing = airbnbData.copy()
Transaction = airbnbData[airbnbData['host_response_rate'] != 0].copy()
Transaction = airbnbData[airbnbData['host_acceptance_rate'] != 0].copy()

XPricing = Pricing[['host_response_rate', 'host_acceptance_rate', 'host_listings_count',
'accommodates', 'room_type', 'bedrooms', 'bathrooms', 'beds', 'cleaning_fee', 'security_deposit',
'minimum_nights', 'maximum_nights', 'number_of_reviews', 'latitude', 'longitude']]
yPricing = Pricing['price']

XTransaction = Transaction[['host_response_rate', 'host_acceptance_rate', 'host_listings_count',
'accommodates', 'room_type', 'bedrooms', 'bathrooms', 'beds', 'cleaning_fee', 'security_deposit',
'minimum_nights', 'maximum_nights', 'number_of_reviews', 'latitude', 'longitude']]
yTransaction = Transaction['price']

XP_train,XP_test,yP_train,yP_test = train_test_split(XPricing,yPricing,test_size = 0.3)
XT_train,XT_test,yT_train,yT_test = train_test_split(XTransaction,yTransaction,test_size = 0.3)

knn = KNeighborsRegressor(n_neighbors = 20)
knn.fit(XP_train,yP_train)

print(knn.predict(XP_test))
print(yP_test)
print(knn.score(XP_test,yP_test))
