import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv("superstore.csv")
print(df.head())
print(df.columns)
print(df.info())
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
sales_data = df.groupby('Order_Date')['Sales'].sum().reset_index()
sales_data['Days'] = range(len(sales_data))
X = sales_data[['Days']]
y = sales_data['Sales']
model = LinearRegression()
model.fit(X, y)
sales_data['Predicted_Sales'] = model.predict(X)
plt.figure(figsize=(10,5))

plt.plot(sales_data['Order_Date'], sales_data['Sales'], label='Actual Sales')

plt.plot(
    sales_data['Order_Date'],
    sales_data['Predicted_Sales'],
    label='Predicted Sales'
)

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecasting")

plt.legend()

plt.show()
sales_data.to_csv("sales_predictions.csv", index=False)