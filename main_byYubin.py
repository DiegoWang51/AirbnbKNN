# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17

@author: Yubin Hu

修改：
1.删减相互相关的特征
2.增加了归一化

目前score在0.6左右

需要：
1.测试经纬度用距离替换效果
2.加权测试

"""

#读入数据
import pandas as pd
import numpy as np
dc_listings=pd.read_csv("dc_airbnb-filtered.csv")

#数据预处理
stripped_commas = dc_listings['price'].str.replace(',', '') #数据预处理,去掉逗号
stripped_dollars = stripped_commas.str.replace('$', '') #数据预处理，去掉美金符号
dc_listings['price'] = stripped_dollars.astype('float') #数据预处理，强制转换为float类型

dc_listings['host_response_rate'] = dc_listings['host_response_rate'].str.replace('%', '') #数据预处理，去掉百分号
dc_listings['host_acceptance_rate'] = dc_listings['host_acceptance_rate'].str.replace('%', '')
stripped_commas = dc_listings['cleaning_fee'].str.replace(',', '') #数据预处理,去掉逗号
stripped_dollars = stripped_commas.str.replace('$', '') #数据预处理，去掉美金符号
stripped_spaces = stripped_dollars.str.replace(' ','') #数据预处理，去掉空格
dc_listings['cleaning_fee'] = stripped_spaces

stripped_commas = dc_listings['security_deposit'].str.replace(',', '') #数据预处理,去掉逗号
stripped_dollars = stripped_commas.str.replace('$', '') #数据预处理，去掉美金符号
stripped_spaces = stripped_dollars.str.replace(' ','')
dc_listings['security_deposit'] = stripped_spaces

dc_listings['room_type'] = dc_listings['room_type'].str.replace('Entire home/apt', '3') #数据预处理,赋值，entire 3， private 2， shared 1
dc_listings['room_type'] = dc_listings['room_type'].str.replace('Private room', '2')
dc_listings['room_type'] = dc_listings['room_type'].str.replace('Shared room', '1')


dc_listings['host_listings_count'] = dc_listings['host_listings_count'].astype('float')
dc_listings['accommodates'] = dc_listings['accommodates'].astype('float')
dc_listings['beds'] = dc_listings['beds'].astype('float')
dc_listings['maximum_nights'] = dc_listings['maximum_nights'].astype('float')
dc_listings['cleaning_fee'] = dc_listings['cleaning_fee'].astype('float')
dc_listings['security_deposit'] = dc_listings['security_deposit'].astype('float')
dc_listings['room_type'] = dc_listings['room_type'].astype('float')
dc_listings['host_response_rate'] = dc_listings['host_response_rate'].astype('float')
dc_listings['host_acceptance_rate'] = dc_listings['host_acceptance_rate'].astype('float')

dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))] #对现有数列随机排列

#数据选择
house_features=dc_listings
house_features=house_features.dropna(subset=['host_acceptance_rate']) #去除未成交房屋
del house_features['city'] #去除重复信息
del house_features['zipcode']
del house_features['state']

#test
del house_features['minimum_nights']
del house_features['maximum_nights']
del house_features['host_listings_count']
#del house_features['cleaning_fee']
#del house_features['security_deposit']
del house_features['bedrooms']
del house_features['beds']
del house_features['host_response_rate']
del house_features['host_acceptance_rate']

del house_features['longitude']
del house_features['latitude']

house_features=house_features.fillna(0) #补充 cleaning fee等列的nan

AirbnbKNN_X = house_features #产生KNN输入
AirbnbKNN_y = np.array(house_features['price'])
del AirbnbKNN_X['price']
AirbnbKNN_X = np.array(AirbnbKNN_X)

print(house_features.iloc[0])

#归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
AirbnbKNN_X = min_max_scaler.fit_transform(AirbnbKNN_X)

#数据切分
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(AirbnbKNN_X,AirbnbKNN_y,test_size = 0.3)

#训练模型
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5) #定义用sklearn中的KNN分类算法 
knn.fit(X_train,y_train)

# 用训练好的模型进行分类
#print(knn.predict(X_test))   #这里的knn就是已经train好了的knn
#print(y_test)    # 对比真实值
print(knn.score(X_test,y_test)) #输出模型准确率 



##print(house_features.iloc[0])
##print(house_features.iloc[0])
##print(AirbnbKNN_X)
##print(AirbnbKNN_y)


#可视化
viz_flag=1
if(viz_flag==True):
    import matplotlib.pyplot as plt
    
#    string='security_deposit'
#    plt.scatter(AirbnbKNN_y,house_features[string])
#    plt.xlabel("Nightly Price")
#    plt.ylabel(string.replace('_', ' ')) 
    
    plt.scatter(knn.predict(X_test),y_test)
    plt.xlabel("Predict")
    plt.ylabel("Correct") 
    
    ##string='accommodates'
    ##plt.scatter(AirbnbKNN_y,house_features[string])
    ##plt.xlabel("Nightly Price")
    ##plt.ylabel(string.replace('_', ' ')) #数据预处理,去掉逗号

    plt.show()