# Phone-price-category-classifier
Can be used to classify price category of phone 

## 📋 Table of Contents
- [About](#about)
- [Demo](#demo)
- [Features](#features)
- [Dataset](#dataset)
- [Model / Approach](#model--approach)


# About 🛒
This is a classifier project which takes input data features of a mobile phone from the user and predicts which price category it can belong to ie Low,Mid,High,Premium.
It also uses 3 different models LogisticRegression , SupporVectorMachine , KNearestNeighbour . So users can also compare the output of these three models and also check their metrics .

# Demo 📄
<img width="1907" height="887" alt="Screenshot 2026-10-01 175819" src="https://github.com/user-attachments/assets/01fe15ad-2760-411b-b37f-7d051a553ac0" />
<img width="1916" height="677" alt="Screenshot 2026-10-01 175837" src="https://github.com/user-attachments/assets/829b4b3b-8f99-45bf-bd4e-8523473a7c3d" />

# Features ⚙️
Features are :
- Used three different classification algorithm , so we could choose any one of them to compare
- predicts the price category based on the input features

# Dataset 📈
- Data set used is :- Link: https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price

# Model/Approach 🧠
- The following steps were taken :-
- Obtaining a dataset and then doing some feature engineering such as converting the str columns into float or int64 do that our model could work upon it (You may not see the algo I used to clean the dataset as I use VS code to i cleaned the dataset and then used the clean dataset .
- Doing feature scaling using standard scaler , specially it was more important for KNN model
- creating three models and tweaking their hyper-parameters(C,penalty,kernel,etc) to get the best optimum results

# Note
This is a project made to compare the working of the three main classification models (not including Naive Bayes) .
You can also tune the hyper-parameters if you like to .
