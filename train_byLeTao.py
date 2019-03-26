import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

import matplotlib.pyplot as plt


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

allFeatures = ['host_response_rate', 'host_acceptance_rate', 'host_listings_count',
'accommodates', 'room_type', 'bedrooms', 'bathrooms', 'beds', 'cleaning_fee', 'security_deposit',
'minimum_nights', 'maximum_nights', 'number_of_reviews', 'latitude', 'longitude']

usedFeatures = ['room_type', 'bedrooms', 'bathrooms', 'beds', 'latitude', 'longitude']

Pricing = airbnbData.copy()
Transaction = airbnbData[airbnbData['host_response_rate'] != 0].copy()
Transaction = Transaction[Transaction['host_acceptance_rate'] != 0].copy()

XPricing = Pricing[usedFeatures]
yPricing = Pricing['price']

XTransaction = Transaction[usedFeatures]
yTransaction = Transaction['price']

scalers = ['Binarizer', 'FunctionTransformer', 'Imputer', 'KernelCenterer',
'MaxAbsScaler', 'MinMaxScaler', 'Normalizer', 'OneHotEncoder', 'QuantileTransformer']

pricingScores = [0 for i in range(len(scalers))]
transactionScores = [0 for i in range(len(scalers))]

for testIteration in range(10):

    scalerIndex = -1

    for scaler in scalers:

        # print('\n', scaler)
        scalerIndex += 1

        # Converts DataFrame to ndarray, losing feature names
        # XPricing = preprocessing.MinMaxScaler().fit_transform(XPricing)
        # XTransaction = preprocessing.MinMaxScaler().fit_transform(XTransaction)
        XPricingS = eval('preprocessing.{}().fit_transform(XPricing)'.format(scaler))
        XTransactionS = eval('preprocessing.{}().fit_transform(XTransaction)'.format(scaler))

        XPS_train,XPS_test,yP_train,yP_test = train_test_split(XPricingS, yPricing, test_size = 0.3)
        XTS_train,XTS_test,yT_train,yT_test = train_test_split(XTransactionS, yTransaction, test_size = 0.3)

        knnP = KNeighborsRegressor(n_neighbors = 20)
        knnP.fit(XPS_train, yP_train)

        # print(knnP.predict(XPS_test[:10]))
        # print(np.array(yP_test[:10]))
        # print(knnP.score(XPS_test, yP_test))
        pricingScores[scalerIndex] += knnP.score(XPS_test, yP_test)

        plt.scatter(knnP.predict(XPS_test), np.array(yP_test))
        plt.title('Prediction on Pricing using Normalization of {}'.format(scaler))
        plt.xlabel('Predicted Pricing Price')
        plt.ylabel('Actual Pricing Price')
        plt.figtext(0.6, 0.8, 'KNN Score: {}'.format(round(knnP.score(XPS_test, yP_test), 3)))
        plt.savefig('scaler graphs/{} Prediction'.format(scaler))
        plt.clf()

        knnT = KNeighborsRegressor(n_neighbors = 20)
        knnT.fit(XTS_train, yT_train)

        # print(knnT.predict(XTS_test[:10]))
        # print(np.array(yT_test[:10]))
        # print(knnT.score(XTS_test, yT_test))
        transactionScores[scalerIndex] += knnT.score(XTS_test, yT_test)

        plt.scatter(knnP.predict(XTS_test), np.array(yT_test))
        plt.title('Prediction on Transaction using Normalization of {}'.format(scaler))
        plt.xlabel('Predicted Transaction Price')
        plt.ylabel('Actual Transaction Price')
        plt.figtext(0.6, 0.8, 'KNN Score: {}'.format(round(knnT.score(XTS_test, yT_test), 3)))
        plt.savefig('scaler graphs/{} Transaction'.format(scaler))
        plt.clf()

    print(transactionScores)

pricingScores = [score / 10 for score in pricingScores]
transactionScores = [score / 10 for score in transactionScores]

plt.bar(range(len(scalers)), pricingScores)
plt.setp(plt.xticks()[1], rotation=30)
plt.xticks(range(len(scalers)), scalers)
plt.title('All Types of Scaler Scores for Picing Prediction')
plt.xlabel('Type of Scalers')
plt.ylabel('Average KNN Scores')
plt.savefig('scaler graphs/all pricing scalers')
plt.clf()

plt.bar(range(len(scalers)), transactionScores)
plt.setp(plt.xticks()[1], rotation=30)
plt.xticks(range(len(scalers)), scalers)
plt.title('All Types of Scaler Scores for Transaction Prediction')
plt.xlabel('Type of Scalers')
plt.ylabel('Average KNN Scores')
plt.savefig('scaler graphs/all transaction scalers')
plt.clf()

print('\n', transactionScores)
