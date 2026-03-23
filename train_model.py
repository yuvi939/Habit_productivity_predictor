import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

data=pd.read_csv("productivity_dataset.csv")

data['productivity'] = data['productivity'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})


x = data[['sleep','exercise','work','breaks']]
y = data['productivity']



model = RandomForestClassifier(n_estimators=7)
model.fit(x,y)

with open("model.pkl","wb") as f:
    pickle.dump(model,f)
