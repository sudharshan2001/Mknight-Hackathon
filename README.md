# Machineknight-Hackathon

## EDA Report:
    There are some null values from the columns extracted from the amenities columns which is imputed (Most Frequent values)
  
    Categorical columns are encoded
  
    From the activation_date we extracted month and year
 
## Feature Engineering and Model Selection
 
    Several tree models and boosting techniques are used such as Random Forest, Decision Tree, AdaBoost, XGBoost and selected the best model
  
    Used GridSearch-CV with 5 CV to tune the hyperparameter of the model,
  
    Filtered out important feautres from it by using RFECV and inbuilt XGBOOST Feature importance method
  
 Selected Features are ['longitude', 'latitude', 'property_size', 'type', 'property_age',
       'cup_board', 'total_floor', 'month','floor', 'lease_type', 'facing', 'furnishing', 'bathroom', 'balconies', 'parking', 'GP','building_type',
          'water_supply', 'negotiable', 'lift']
  
  
  ![featureimp](https://user-images.githubusercontent.com/72936645/188326501-97dfd176-b63d-4280-9d18-ba16b96ad54d.png)

  
| Model                  | RMSE           | 
|:-----------------------|:---------------|
| Random Forest          | 4218           | 
| AdaBoost RF            | 4125           |
| Decision Tree          | 4863           | 
| AdaBoost DT            | 4233           |
| XGBoost                | 4100           | 
| AdaBoost XGBoost       | 3957           |


Download this folder 

```pip install requirements.txt```

``` python app.py ```
