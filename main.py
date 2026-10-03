import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#Importing the data and removing low contrtibuting columns
data = pd.read_csv(r"price.csv")
data = data.drop(columns=["Model"])

#Making train test split
X= data.drop(columns=["price"])
Y= data["price"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=17)

#Encoding output column
le = LabelEncoder()
y_train_encoded =  le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

#Making all functions and models
scaler = StandardScaler()
onehot = OneHotEncoder(sparse_output=False,drop="first")
model1=LogisticRegression(l1_ratio=1,solver='saga',C=0.1,max_iter=1000)
model2=SVC(kernel="rbf",C=1)
model3=KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
model4=DecisionTreeClassifier(max_depth=5,criterion="gini")

#Transformers
c1 = ColumnTransformer(transformers=[
    ("onehot",onehot,[0])
],remainder="passthrough")
c2 = ColumnTransformer(transformers=[
    ("standard scaling",scaler,[i for i in range(22)])
],remainder="passthrough")



#pipeline
pipe_common = Pipeline([
    ("c1",c1),
    ("c2",c2)
])
m1 = Pipeline([
    ("pipe_common",pipe_common),
    ("model",model1)
])
m2 = Pipeline([
    ("pipe_common",pipe_common),
    ("model",model2)
])
m3 = Pipeline([
    ("pipe_common",pipe_common),
    ("model",model3)
])
m4 = Pipeline([
    ("pipe_common",pipe_common),
    ("model",model4)
])


