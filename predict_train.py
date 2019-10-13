import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

X = pd.read_csv('data/train.csv', index_col='Id')

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice

X.drop(axis=1, columns=['SalePrice'], inplace=True)

categorical_cols = [col for col in X.columns
                    if X[col].dtype in ['object']]

numerical_cols = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                  'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF',
                  'TotalBsmtSF',
                  'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                  '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                  'BedroomAbvGr',
                  'KitchenAbvGr', 'TotRmsAbvGrd',
                  'Fireplaces',
                  'GarageCars', 'GarageArea',
                  'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                  'MiscVal', 'MoSold', 'YrSold',
                  'PoolArea']

null_numerical_cols = ['LotFrontage', 'GarageYrBlt']

# Keep selected columns only
my_cols = categorical_cols + numerical_cols + null_numerical_cols

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')
null_numerical_transformer = SimpleImputer(strategy='constant', fill_value=0)

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('null_num', null_numerical_transformer, null_numerical_cols),
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train_final = preprocessor.fit_transform(X[my_cols])
# X_valid_final = preprocessor.transform(X_valid[my_cols])

xg_model = XGBRegressor(random_state=1,
                        n_estimators=1000,
                        learning_rate=0.05,
                        objective='reg:squarederror',
                        n_jobs=8)

# xg_model.fit(X_train_final,
#              y_train,
#              early_stopping_rounds=100,
#              eval_set=[(X_valid_final, y_valid)],
#              verbose=False)

# prediction = xg_model.predict(X_valid_final)

kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(xg_model, X_train_final, y,
                              cv=kfold,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("MAE scores mean:\n", scores.mean())
print("MAE scores std:\n", scores.std())

# print(mean_absolute_error(prediction, y_valid))
