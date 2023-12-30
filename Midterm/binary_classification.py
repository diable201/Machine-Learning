from catboost import CatBoostClassifier
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop('target', axis=1)
y = train['target']

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    verbose=False,
    depth=7
)

model.fit(X, y, [])

x_test = test.drop(["Id"], axis=1)
test['target'] = (model.predict_proba(x_test)[:, 1] > 0.95).astype(int)

answers = test[['target', 'Id']]
answers.to_csv('answers.csv', index=False)
