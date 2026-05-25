import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df =pd.read_csv('superstore.csv')
print(df.head())
print(df.info())
print(df.columns)
print(df.isnull().sum())
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)
df.fillna("Unknown", inplace=True)
df.drop_duplicates(inplace=True)
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
''' sales by region'''
region_sales = df.groupby('Region')['Sales'].sum()
plt.figure(figsize=(7,7))
plt.pie(
    region_sales,
    labels=region_sales.index,
    autopct='%1.1f%%'
)
plt.title("Sales Distribution by Region")
plt.show()
#profit trend
profit_trend = df.groupby('Order_Date')['Profit'].sum()
profit_trend.plot()
plt.title("Profit Trend")
plt.xlabel("Date")
plt.ylabel("Profit")
plt.show()
#sales over trend
plt.subplot(3, 2, 1)
df['Month'] = df['Order_Date'].dt.strftime('%b')
sales_trend = df.groupby('Month')['Sales'].sum()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sales_trend = sales_trend.reindex(month_order)
plt.plot(sales_trend)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
#category distribution
plt.subplot(3, 2, 3)
category_sales = df.groupby('Category')['Sales'].sum()
plt.pie(
    category_sales,
    labels=category_sales.index,
    autopct='%1.1f%%'
)
plt.title("Category Distribution")
#segment distribution
plt.subplot(3, 2, 4)

segment_sales = df.groupby('Segment')['Sales'].sum()

plt.pie(
    segment_sales,
    labels=segment_sales.index,
    autopct='%1.1f%%'
)

# Donut circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Segment Distribution")
#profit vs sales
plt.subplot(3, 2, 5)
plt.scatter(df['Sales'], df['Profit'])
plt.title("Profit vs Sales")
plt.xlabel("Sales")
plt.ylabel("Profit")
#correlation heatmap
plt.subplot(3, 2, 6)
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
print("\nSummary Statistics:")
print(df.describe())
print("Automation completed successfully")

'''df.to_csv("cleaned_superstore.csv", index=False)'''