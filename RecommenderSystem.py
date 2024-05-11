#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mlxtend


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from mlxtend.frequent_patterns import apriori, association_rules


# In[3]:


file_path = r'C:\Users\lenovo\Desktop\online_retail_II.csv'
df = pd.read_csv(file_path)


# In[4]:


#Data PreProcessing


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe().T


# In[10]:


#Exploratory Data Analysis (EDA)


# In[11]:


#regulating
df.columns = [col.replace(" ", "_").upper() for col in df.columns]
df.dropna(inplace=True)


# In[12]:


#Handling Outliers
df.describe().T


# In[13]:


qtt = df.loc[df["QUANTITY"]<0,"QUANTITY"].count()
inv = df["INVOICE"].str.contains("C",na=False).sum()


# In[14]:


df.drop(df[df["INVOICE"].str.contains("C")].index,inplace=True)


# In[15]:


qtt = df.loc[df["QUANTITY"]<0,"QUANTITY"].count()
inv = df["INVOICE"].str.contains("C",na=False).sum()
print(f"The number of negative QUANTITY values: {qtt}\nThe number of INVOICES containing 'C' : {inv}")


# In[16]:


df.describe().T


# In[17]:


def handling_outlier(data,variable):
    quartile1 = data[variable].quantile(0.01) # Range (%1-%99)
    quartile3 = data[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    data.loc[data[variable] < low_limit, variable] = low_limit
    data.loc[data[variable] > up_limit, variable] = up_limit


# In[18]:


handling_outlier(df,"QUANTITY")
handling_outlier(df,"PRICE")



# In[19]:


df.describe().T


# In[20]:


# Selecting some countries from the data set
list_cntry = ["Greece","Singapore","Netherlands","Switzerland","Cyprus","France","Korea","Canada"]
for number,country in enumerate(list_cntry):
    list_cntry[number] = df[df['COUNTRY'] == country]


# In[21]:


del df


# In[22]:


df = pd.concat(list_cntry,axis=0)
df = df.sort_index()


# In[23]:


df = df.reset_index(drop=True)
df.shape


# In[24]:


#Data Analysis & Visualization


# In[25]:


# Top 10 best selling products

product_count = df.groupby("DESCRIPTION")["QUANTITY"].sum().nlargest(10)
product_count=product_count.reset_index()


# In[26]:


plt.figure(figsize=(12, 8))

ax = sns.barplot(data=product_count,y="DESCRIPTION",x="QUANTITY",palette="icefire")

for i in ax.containers:
    ax.bar_label(i,)

ax.set_title("Top 10 Best Selling Products")
plt.xlabel("Total Quantity")
plt.ylabel("Products")
plt.tight_layout()
plt.show()


# In[27]:


# Price of any product by country

list_country, list_price = [], []
for col in df["COUNTRY"].unique():
    price = df.loc[(df["COUNTRY"] == col) & (df["DESCRIPTION"] == "WHITE HANGING HEART T-LIGHT HOLDER"), "PRICE"].mean()

    list_country.append(col)
    list_price.append(round(price,3))

df_price = pd.DataFrame(columns=["COUNTRY"],data=list_price,index=list_country)
df_price.dropna(inplace=True)
df_price = df_price.sort_values(by="COUNTRY",ascending=False)


# In[28]:


plt.figure(figsize=(12, 8))

ax = sns.barplot(data=df_price,y=df_price.index,x="COUNTRY",palette="rocket_r")

for i in ax.containers:
    ax.bar_label(i,)

ax.set_title("Price of 'White Hanging Heart t-light Holder' by Country")
plt.xlim(0, 4)
plt.xlabel("Unit Price")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


# In[29]:


# Total amount of the first 10 products

df["TOTAL_AMOUNT"] = df["QUANTITY"] * df["PRICE"]
df.head()

total_amount = df.groupby("DESCRIPTION")["TOTAL_AMOUNT"].sum().nlargest(10)
total_amount=total_amount.reset_index()


# In[30]:


plt.figure(figsize=(12, 8))

ax = sns.barplot(data=total_amount,y="DESCRIPTION",x="TOTAL_AMOUNT",palette="viridis")

for i in ax.containers:
    ax.bar_label(i,)

ax.set_title("Total Amount of the First 10 Products")
plt.xlabel("Total Amount")
plt.ylabel("Products")
plt.tight_layout()
plt.show()


# In[31]:


#Preparing the ARL Data Structure


# In[32]:


# Reaching the product quantities in each invoice.
df.groupby(["INVOICE","DESCRIPTION"])["QUANTITY"].sum().head(20)

# if you want you can use this code, it gives same result
# df.groupby(["INVOICE","DESCRIPTION"]).agg({"QUANTITY":"sum"}).head(20)


# In[33]:


# Sorting descriptions by columns
df.groupby(["INVOICE", "DESCRIPTION"]).agg({"QUANTITY": "sum"}).unstack().iloc[0:5, 0:5]


# In[34]:


# Filling nan values with zero
df.groupby(['INVOICE', 'DESCRIPTION']).agg({"QUANTITY": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]


# In[35]:


# 0,0 is converted to 0, if there is a value then it is 1
df.groupby(['INVOICE', 'DESCRIPTION']).agg({"QUANTITY": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


# In[36]:


# Changing product names with stock code
df.groupby(['INVOICE', 'STOCKCODE']).agg({"QUANTITY": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


# In[37]:


# It ready !!
df_arl = df.groupby(['INVOICE', 'STOCKCODE']).agg({"QUANTITY": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


# In[38]:


# Finding product name from stock code
def prdct_name_finder(data,stckcde):
    product_name = data[data["STOCKCODE"] == stckcde][["DESCRIPTION"]].values[0].tolist()
    print(product_name)


# In[39]:


prdct_name_finder(df,"85014A")


# In[40]:


#Association Rule Analysis


# In[41]:


frequent_itemsets = apriori(df_arl,min_support=0.01,use_colnames=True)


# In[42]:


frequent_itemsets.sort_values("support", ascending=False)


# In[43]:


rules = association_rules(frequent_itemsets,metric="support",min_threshold=0.01)


# In[44]:


# Filtering
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]


# In[45]:


# Filtering by confidence
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)].sort_values("confidence", ascending=False)


# In[46]:


#Application


# In[47]:


def prdct_name_finder(data,stckcde):
    product_name = data[data["STOCKCODE"] == stckcde][["DESCRIPTION"]].values[0].tolist()
    return product_name


# In[48]:


def arl_recommender(rules_df, product_id, rec_count):
    
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    recommendation_list_name = []
    
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j[1] == product_id:
                for k in list(sorted_rules.iloc[i]["consequents"]):
                    if k[1] not in recommendation_list:
                        recommendation_list.append(k[1])
    added_product = prdct_name_finder(df,product_id)
    print(f"Added to Cart:           {added_product[0]}\n\n")
    print(f"Members Who Bought This Also Bought:\n\n")
    for i in range(0,rec_count):
        recommendation_list_name.append(prdct_name_finder(df,recommendation_list[i]))
        print(f"                         {recommendation_list_name[i][0]}\n")


# In[49]:


arl_recommender(rules, "85123A", 3)

