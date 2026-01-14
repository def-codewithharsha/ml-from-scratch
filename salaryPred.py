import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Linear_Regression import LinearRegression as LR

salary_df=pd.read_csv('Salary_Data.csv',usecols=["Years of Experience","Salary"])
print(salary_df.head())
df_cleaned=salary_df.dropna()
 
print("Missing values:\n", df_cleaned.isnull().sum())


#spliting feature and target Y=wX+b Y is dependent on x (salary depend on experience)
X=df_cleaned[["Years of Experience"]].values
Y=df_cleaned["Salary"].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=2)

#Feature Scaling(# Scale features and target)

X_scalar=StandardScaler()
Y_scalar=StandardScaler()

X_train_scaled=X_scalar.fit_transform(X_train)
Y_train_scaled=Y_scalar.fit_transform(Y_train.reshape(-1,1)).flatten()

#training Linear Regression Model
model=LR(learning_rate=0.001,num_of_iterations=5000)
model.fit(X_train_scaled,Y_train_scaled)

#just to check weight and bais
print("weight",model.w)
print('bais',model.b)

#predict test data
X_test_scaled=X_scalar.transform(X_test)
Y_pred_scaled=model.predict(X_test_scaled)
Y_pred =Y_scalar.inverse_transform(Y_pred_scaled.reshape(-1,1)).flatten()

#Plot

plt.scatter(range(len(Y_test)), Y_test, color='red', label='Actual')
plt.scatter(range(len(Y_pred)), Y_pred, color='blue', label='Predicted')
plt.xlabel("Test Sample Index")
plt.ylabel("Salary")
plt.title("Age + Experience vs Salary (Custom Linear Regression)")
plt.legend()
plt.show()
