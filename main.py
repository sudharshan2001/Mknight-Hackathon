import numpy as np 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import json
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv("./hackathon/hackathon/data/train.csv")

dict_columns = pd.DataFrame([(json.loads(df.amenities.loc[i])) for i in range(len(df))])

dict_columns = dict_columns.replace({True: 1, False: 0})

df.drop(["amenities"], inplace=True, axis=1)

df = pd.concat([df, dict_columns], axis=1)

obj_names=["type","lease_type","furnishing","parking","facing","water_supply","building_type"]
obt_type = df[obj_names]

encode1 = OrdinalEncoder()
encoder = encode1.fit(obt_type)
encoded = encoder.transform(obt_type)

df[["type","lease_type","furnishing","parking","facing","water_supply","building_type"]] = encoded

df["year"] = pd.to_datetime(df['activation_date'] , format="%d-%m-%Y %H:%M").dt.year
df["month"] = pd.to_datetime(df['activation_date'] , format="%d-%m-%Y %H:%M").dt.month

locality_var = df['locality']
df.drop(["locality", "GYM"], axis=1, inplace=True)

cate_names=["CLUB","CPA","SERVANT","GP","RWH","STP","VP"]
cate_type = df[cate_names]

imp_freq = SimpleImputer(strategy='most_frequent')

imp_freq_cate = imp_freq.fit(cate_type)
transformed_cate_type = imp_freq_cate.transform(cate_type)

df[cate_names] = transformed_cate_type

actication_date = df.activation_date
df.drop(["activation_date","id"],axis=1, inplace=True)
df.drop(["LIFT","POOL"],axis=1, inplace=True)

X = df.drop('rent',axis=1)
y = df['rent']

features=['longitude', 'latitude', 'property_size', 'type', 'property_age',
       'cup_board', 'total_floor', 'month','floor', 'lease_type', 'facing', 'furnishing', 'bathroom', 'balconies', 'parking', 'GP','building_type', 'water_supply', 'negotiable', 'lift']

temp_x = X[features]
New_X_train, New_X_val, New_y_train, New_y_val = train_test_split(temp_x,y, test_size=0.9, random_state=0)

regressor=xgb.XGBRegressor(eval_metric='rmse')

param_grid1 = {"max_depth":    [int(x) for x in np.linspace(5, 20, num = 4)],
              "n_estimators": [int(x) for x in np.linspace(200, 800, num = 10)],
              "learning_rate": [0.01, 0.02],}


search = GridSearchCV(regressor, param_grid1, cv=5,verbose=2).fit(New_X_train, New_y_train)

xgb_model = xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],  # 0.02
                           n_estimators  = search.best_params_["n_estimators"],  # 733
                           max_depth     = search.best_params_["max_depth"],  # 5
                           eval_metric='rmse')

xgb_model.fit(New_X_train, New_y_train)

print(mean_squared_error(New_y_val, xgb_model.predict(New_X_val), squared=False))

pickle.dump(regressor, open('model.sav', 'wb'))
