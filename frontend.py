import main
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#Creating test/train data
x_train = main.x_train
x_test = main.x_test
y_train_e = main.y_train_encoded
y_test_e = main.y_test_encoded


col1, col2, col3 = st.columns([1, 4, 1], border=False)

with col1:
    st.markdown(
        """
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            padding-top: 10px;
        ">
            <img
                src="https://media.tenor.com/JU2j5WBbFvEAAAAj/cart.gif"
                style="
                    width: 90px;
                    height: 90px;
                    object-fit: contain;
                "
            >
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <h1 style="
            text-align: center;
            color: #6366F1;
            font-size: 42px;
            font-weight: 700;
            margin: 0;
            white-space: nowrap;
        ">
            Mobile Price Category Classifier
        </h1>
        """,
        unsafe_allow_html=True
    )


model = st.container(border=True)
with model:
    model_select=st.selectbox(":green[Enter using which model you] :blue[want the] :red[prediction]",["Logistic Regression","Support Vector machine","K-nearest neighbour"])

body_col1,body_col2 = st.columns(2)

with body_col1:
    storage = st.number_input(":orange[Enter the storage in GB]",min_value=1,value=128)
    ram = st.number_input(":orange[Enter the Ram in GB]",min_value=1,value=8)
    ss = st.number_input(":orange[Enter the screen size in inches]",min_value=1.0,value=5.0,step=0.01)
    battery = st.number_input(":orange[Enter the battery capacity in mAh]",min_value=500,value=3000)

with body_col2:
    c1 = st.number_input(":blue[Enter the main Camera quality in MP]",min_value=0,value=0)
    c2 = st.number_input(":blue[Enter 2nd Camera quality in MP]",min_value=0,value=0)
    c3 = st.number_input(":blue[Enter 3rd Camera quality in MP]",min_value=0,value=0)
    c4 = st.number_input(":blue[Enter 4th Camera quality in MP]",min_value=0,value=0)
    st.write(":red[Enter 0 if the phone doesnot contain any of the following camera]")

brand = st.container(border=True)

with brand:
    phone_brand = st.selectbox(":violet[Enter the brand of mobile phone]",["Samsung","Xiaomi","Oppo","Realme","Vivo","Apple","Nokia","Motorola","OnePlus","Huawei","Google","Asus","LG","Blackberry","Sony"])
    metric = st.checkbox(":yellow[Do you want to see the metrics of the selected model ?]")
    submit = st.button("Enter")

#3premium 1low 2mid 0high

result = st.container()

list1=[]
list1.append(phone_brand)
list1.append(storage)
list1.append(ram)
list1.append(ss)
list1.append(battery)
list1.append(c1)
list1.append(c2)
list1.append(c3)
list1.append(c4)
df= pd.DataFrame([list1],columns=main.X.columns)

if(submit==True):
    if(model_select=="Logistic Regression"):
        main.m1.fit(x_train,y_train_e)
        y_pred_metric = main.m1.predict(x_test)
        y_pred_test = main.m1.predict(df)
    elif(model_select=="Support Vector machine"):
        main.m2.fit(x_train,y_train_e)
        y_pred_metric = main.m2.predict(x_test)
        y_pred_test = main.m2.predict(df)
    else:
        main.m3.fit(x_train,y_train_e)
        y_pred_metric = main.m3.predict(x_test) 
        y_pred_test = main.m3.predict(df)
       
    with result:  
        if(y_pred_test[0]==0):
            st.success(f"The Price category Prediction made by {model_select} model is High")
        elif(y_pred_test[0]==1):
            st.success(f"The Price category Prediction made by {model_select} model is Low")    
        elif(y_pred_test[0]==2):
            st.success(f"The Price category Prediction made by {model_select} model is Mid")    
        else:
            st.success(f"The Price category Prediction made by {model_select} model is Premium")
    
        if(metric):
            st.write(f":orange[The Accuracy we achieved during training is {accuracy_score(y_test_e,y_pred_metric)}]")
            st.write(f":orange[The Precision we achieved during training is {precision_score(y_test_e,y_pred_metric,average="macro")}]")
            st.write(f":blue[The Recall we achieved during training is {recall_score(y_test_e,y_pred_metric,average="macro")}]")
            st.write(f":blue[The F1 we achieved during training is {f1_score(y_test_e,y_pred_metric,average="macro")}]")