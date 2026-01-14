import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

from MyLogisticRegression import MyLogisticRegression as LogReg

def main():

    CSV_PATH = "breast-cancer.csv"
    TARGET_COLUMN = "diagnosis"

    df= pd.read_csv(CSV_PATH)

    df= df.dropna()

    df[TARGET_COLUMN]= df[TARGET_COLUMN].map({"M":1,"B":0})
    

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Label/Target column not found in the dataset")
       

    X = df.drop(columns=[TARGET_COLUMN,"id"]).values
    Y = df[TARGET_COLUMN].values

    scaler=StandardScaler()

    X = scaler.fit_transform(X)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    model=LogReg(learning_rate=0.1,num_of_iterations=5000)

    model.fit(X_train,Y_train)

    Y_pred=model.predict(X_test)

    accuracy = accuracy_score(Y_test,Y_pred)
    cm=confusion_matrix(Y_test,Y_pred)

    print("\n Model Evaluation :")
    print("Accuracy: ",accuracy)
    print("Confusion Matrix :",cm)


main()
        



