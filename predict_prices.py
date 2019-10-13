import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

train_set = pd.read_csv('data/train.csv', index_col='Id')
test_set = pd.read_csv('data/test.csv', index_col='Id')

train_set.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_set.SalePrice

train_set.drop(axis=1, columns=['SalePrice'], inplace=True)

categorical_cols = [col for col in train_set.columns if train_set[col].dtype in ['object']]

numerical_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                  'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                  'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'PoolArea']

X_train, X_valid, y_train, y_valid = train_test_split(train_set, y, random_state=1, test_size=0.2)

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_test = test_set[my_cols]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train_final = preprocessor.fit_transform(X_train[my_cols])
X_valid_final = preprocessor.transform(X_valid[my_cols])
X_test_final = preprocessor.transform(X_test)
X_test_final.index = X_test.index

xg_model = XGBRegressor(random_state=1, n_estimators=1000, learning_rate=0.05)

xg_model.fit(X_train_final, y_train)

prediction = xg_model.predict(X_test_final)

output = pd.DataFrame({'Id': X_test_final.index,
                       'SalePrice': prediction})
output.to_csv('data/submission.csv', index=False)

# print(mean_absolute_error(prediction, y_valid))
