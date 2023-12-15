#!/usr/bin/env python
# coding: utf-8

# # AeroFit Business Case Study

# ##### Business Problem
# 
# The market research team at AeroFit wants to identify the characteristics of the target audience for each type of
# treadmill offered by the company, to provide a better recommendation of the treadmills to the new customers. The team
# decides to investigate whether there are differences across the product with respect to customer characteristics.
# 
# Perform descriptive analytics to create a customer profile for each AeroFit treadmill product by developing
# appropriate tables and charts. For each AeroFit treadmill product, construct two-way contingency tables and compute
# all conditional and marginal probabilities along with their insights/impact on the business.

# ### Importing libraries

# In[392]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# #### Bar plot formate

# In[393]:


def show_values_on_bars(axs, h_v="v", space=1):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# ### Reading file and preparing window

# In[394]:


df = pd.read_csv('Aerofit_treadmill.csv')
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 120)
pd.set_option('display.max_row', 4000)


# ### Data parameters
# 
# * Product Purchased : KP281, KP481, or KP781
# * Age : In years
# * Gender : Male/Female
# * Education : In years
# * MaritalStatus : Single or partnered
# * Usage : The average number of times the customer plans to use the treadmill each week.
# * Income : Annual income (in USD)
# * Fitness : Self-rated fitness on a 1-to-5 scale, where 1 is the poor shape and 5 is the excellent shape.
# * Miles : The average number of miles the customer expects to walk/run each week
# 

# ### getting starter information
# 
# * data shape
# * data info
# * data head
# * data tail
# * missing values
# * data description
# * check duplicates
# * drop duplicates
# * drop unnecessary columns
# 

# In[395]:


cs_name = 'Aerofit case study'
print(f'{cs_name}, shape is {df.shape}')
print()
print()
print(f"{cs_name} basic information")
print()
print(df.info())
print()
print()
print(f"{cs_name} Null value count percentage:")
print()
print(df.isnull().sum(axis=0) / len(df) * 100)
print()
print()
print(f"{cs_name} Description:")
print()
print(df.describe())
print()
print()
print(f"{cs_name} Deep Description:")
print()
print(df.describe(include='object').T)
print()
print()
print(f"{cs_name} Duplicate values:")
print()
print(df.loc[df.duplicated()])


# ### Preparing data (adding and modifying columns)

# In[ ]:





# #### Product price according to the models
# 
# Product Portfolio
# * The KP481 is for mid-level runners that sell for \$ 1,750.
# * The KP281 is an entry-level treadmill that sells for \$ 1,500.
# * The KP781 treadmill is having advanced features that sell for \$ 2,500.
# 

# In[396]:


product_price = pd.DataFrame({
 "Product":["KP281","KP481","KP781"],
 "Product_price":[1500,1750,2500]
 })

product_price


# #### merging product prices in AeroFit data

# In[397]:


df = df.merge(product_price, on="Product", how="left")


# #### fitness by category

# In[398]:


df["Fitness_category"] = df['Fitness']


# In[399]:


df["Fitness_category"].replace({1:"Poor Shape",
 5:"Excellent Shape",
4:"Good Shape",
3:"Average Shape",
2:"Bad Shape"},inplace=True)


# #### Miles per day of usage

# In[400]:


df['Miles per 1 use'] = df['Miles']/df['Usage']


# ### getting starter information after pre-processing the data
# 
# * data shape
# * data info
# * data head
# * data tail
# * missing values
# * data description
# * check duplicates
# * drop duplicates
# * drop unnecessary columns
# 

# In[401]:


cs_name = 'Aerofit case study'
print(f'{cs_name}, shape is {df.shape}')
print()
print()
print(f"{cs_name} basic information")
print()
print(df.info())
print()
print()
print(f"{cs_name} Null value count percentage:")
print()
print(df.isnull().sum(axis=0) / len(df) * 100)
print()
print()
print(f"{cs_name} Description:")
print()
print(df.describe())
print()
print()
print(f"{cs_name} Deep Description:")
print()
print(df.describe(include='object').T)
print()
print()
print(f"{cs_name} Duplicate values:")
print()
print(df.loc[df.duplicated()])


# ##### Observation:
# 
# * Median Age of Customer is 26 years
# * Top selling product is KP281 with highest number of unites sold(80)
# * Maximum users are Married Men.
# * Average fitness level of the customers is 3
# * Average miles for give cutomers' is 94 Miles
# * Average income of the customers is \$ 50596.5.
# * On an average, customers us the tredmill 3 days per week.
# * Highest mile covered in a single use by a customer is 90. (and after a deep analysis we found out, that its a Male customer)
# 

# ## Customer Analysis

# ### Gender distribution of customers

# In[444]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,5))
types = sns.countplot(data = df , x = "Gender") 
plt.xlabel("Gender", fontsize=17)
plt.ylabel("Count",fontsize=17)
plt.title("Gender distribution",fontdict ={"fontsize": 17})
show_values_on_bars(types,h_v="v",space=1)


# In[ ]:





# ### Marital Status based Customer distribution

# In[403]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,5))
types = sns.countplot(data = df , x = "MaritalStatus")
plt.xlabel("Marital Status", fontsize=17)
plt.ylabel("Count",fontsize=17)
plt.title("Marital Status based Customer distribution",fontdict ={"fontsize": 17})
show_values_on_bars(types,h_v="v",space=1)


# ### Age distribution

# In[404]:


age_group_df = df.groupby(['Age'])['Income'].nunique().reset_index()
plt.figure(figsize=(10,5))
sns.set(font_scale = 1.1)
types = sns.lineplot(data = age_group_df , x = "Age", y="Income")
plt.xlabel("Age", fontsize=17)
plt.ylabel("Number of People",fontsize=17)
show_values_on_bars(types,h_v="v",space=1)


# In[405]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,5))
plt.xlabel("Gender",fontsize=17)
plt.ylabel("Age",fontsize=17)
sns.boxplot(x= 'Gender', y='Age', data=df)
plt.show()


# In[ ]:





# ### Miles based cutsomers distribution

# In[406]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,5))
plt.xlabel("Gender",fontsize=17)
plt.ylabel("Density",fontsize=17)
sns.distplot(df["Miles"])
plt.title("Miles based cutsomers distribution",fontdict ={"fontsize": 17})
plt.show()


# In[407]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,5))
plt.xlabel("Gender",fontsize=17)
plt.ylabel("Miles",fontsize=17)
sns.boxplot(x= 'Gender', y='Miles', data=df)
plt.title("Miles and Gender based Customer distribution",fontdict ={"fontsize": 17})
plt.show()


# In[ ]:





# ### Checking for Outliers in Miles

# In[462]:


IQR = np.percentile(df["Miles"],75) - np.percentile(df["Miles"],25)
Q3 = np.percentile(df["Miles"],75)
Q1 = np.percentile(df["Miles"],25)
UpperWhisker = Q3 + (1.5*(IQR))
UpperWhisker

print(f"Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")

outlier_data = df[df["Miles"]>UpperWhisker]
print("Outliers : ",len(outlier_data))

print()
print("Customers who run more than 187.875 (outliers).")
outlier_data["Product"].value_counts()


# ### Income based Gender distribution

# In[408]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,5))
plt.xlabel("Gender",fontsize=17)
plt.ylabel("Density",fontsize=17)
sns.distplot(df["Income"])
plt.title("Income based Customer distribution",fontdict ={"fontsize": 17})
plt.show()


# In[463]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,6))
plt.xlabel("Gender",fontsize=17)
plt.ylabel("Income",fontsize=17)
sns.boxplot(x= 'Gender', y='Income', data=df)
plt.title("Income based Gender distribution",fontdict ={"fontsize": 17})
plt.show()


# ### Checking for Outliers in customer Income

# In[479]:


IQR = np.percentile(df["Income"],75)-np.percentile(df["Income"],25)
Q3 = np.percentile(df["Income"],75)
Q1 = np.percentile(df["Income"],25)
UpperWhisker = Q3 + (1.5*(IQR))
UpperWhisker
print(f"Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")
(df["Income"] > UpperWhisker).value_counts()


# ##### Observation:
# * there are 19 customers, who have Income way above the other customers in the data

# In[ ]:





# ## Product Analysis

# ### Product distribution

# In[410]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,5))
types = sns.countplot(data = df , x = "Product")
plt.xlabel("Product", fontsize=17)
plt.ylabel("Count",fontsize=17)
show_values_on_bars(types,h_v="v",space=1)


# ### Revenue base product distribution

# In[411]:


product_revenue = df.groupby(['Product'])['Product_price'].sum().reset_index().rename(columns=
                                                                                      {'Product':"Product", 
                                                                                       "Product_price":'Product_revenue'})

product_revenue


# In[412]:


sns.set(font_scale = 1.1)
plt.figure(figsize=(10,5))
plt.xlabel("Product",fontsize=17)
plt.ylabel("Product Revenue",fontsize=17)
sns.barplot(x= 'Product', y='Product_revenue', data= product_revenue)
plt.title("Revenue base product distribution",fontdict ={"fontsize": 17})
plt.show()


# ##### Observation:
# 
# * KP281 is the top selling product and KP781 is the lowest selling product
# * KP281 is the highest revenue generating product
# * unlike the huge diffrence in number of product sold between KP481 and KP781 which was 20 (33%), the diffrence betwwen thier revenue is only 5000 USD (4%)
# 

# In[ ]:





# ### Age base product distribution

# In[413]:


product_group_df = df.groupby(['Product']).agg({'Age': 'value_counts'}).rename(columns={'Product':'Product','Age': 'Age','Age':'COUNT'}).reset_index()


# In[443]:


sns.set(style="darkgrid")
plt.figure(figsize=(15,7))
sns.kdeplot(kp281["Age"], shade=True, color="r")
sns.kdeplot(kp481["Age"], shade=True, color="b")
sns.kdeplot(kp781["Age"], shade=True, color="y")
plt.xlabel("AGE",fontsize=20)
plt.ylabel("COUNT",fontsize=20)
plt.legend(["KP281","KP481", "KP781"],loc=1, prop={'size': 17})
plt.title("Age base product distribution",fontdict ={"fontsize": 17})
plt.show()


# In[ ]:





# ## Analysis using Multiple factors 

# ### Gender based product distribution

# In[417]:


pd.crosstab(df['Product'], df['Gender'], margins=True)


# In[418]:


pd.crosstab(df['Product'], df['Gender']).plot(kind="bar", stacked=False, rot=0)
plt.title("Gender based product distribution")
plt.legend()
plt.show()


# In[419]:


(pd.crosstab(df['Product'], df['Gender'], margins=True)/180)*100


# In[ ]:





# #### Observation
# 
# * Gender distribution in KP281 is equal 22.22%
# * In KP481, percentage of Male customers are by 1.1%. 
# * in PK781, percentage of Male customers are by 14.5%. And from this we can say that, when it comes to KP781, its the least favorite product of Female customers and on the other hand 2nd most sold product for Male cutsomers.
# 
# * Marginal Probability for each product is: 
#                                             KP281 = 44.4 %
#                                             KP481 = 33.3 %
#                                             KP781 = 22.2 %
#                                             
# * Marginal Probability for each Gender is:  
#                                             Male = 57.7 %
#                                             Female = 42.2 %
#                                            
# 

# In[420]:


print("Probability that a perticual gender will buy a perticular product shown below:")

(pd.crosstab(df['Product'], df['Gender'], margins=True, normalize='columns'))*100


# ##### Observation:
# 
# * When it comes to KP281 and KP481, it is more likely that a Female customer will buy it. And KP781 on the other hand is more in demand with Male customers as compare to the Female cutomers for that same product.
# 

# ### Marital status based product diestribution

# In[421]:


pd.crosstab(df['Product'],df['MaritalStatus'], margins=True)


# In[422]:


pd.crosstab(df['Product'],df['MaritalStatus']).plot(kind="bar", stacked=False, rot=0)
plt.title("Marital Status based product distribution")
plt.legend()
plt.show()


# In[423]:


(pd.crosstab(df['Product'],df['MaritalStatus'], margins=True)/180)*100


# In[424]:


pd.crosstab(df['Product'],[df['MaritalStatus'],df['Gender']], margins=True)


# In[425]:


pd.crosstab(df['Product'],[df['MaritalStatus'],df['Gender']]).plot(kind="bar", stacked=False, rot=0)
plt.title("Marital Status and gender based product distribution")
plt.legend()
plt.show()


# ##### Obsrvation:
# 
# * Marginal Probability for   
#         Married Customers : 59.44 %
#         Single Customers : 40.555 %
# 
# * there is a 18.9% higher chance of a cutomer being partnered as compare to being single.
# * KP281 is most popular among Partnered Females
# * KP481 is most popular among Partnered Males
# * KP781 is most popular among Partnered Males
# 

# In[426]:


print('Probability of a customer being single or partnered according to the perticular product they perchase is given below')

(pd.crosstab(df['Product'],df['MaritalStatus'], margins=True, normalize='columns'))


# ### Education based Product distribution

# In[427]:


pd.crosstab(df['Education'], df['Product'], margins=True)


# In[428]:


pd.crosstab(df['Education'], df['Product']).plot(kind="bar", stacked=False, rot=0)
plt.title("Education based product distribution")
plt.legend()
plt.show()


# ##### Observation:
# 
# * Product KP281 and KP481 is more popular among cutomers who have education between 14 to 16
# * product KP781 is purchased by customers of education 16 and above.
# 
# * This is because Education and Income have a positive and high correlation, and Product price and Income also have a positive and high correlation. That is the main reason KP781 which is a higher varient of the tredmill with higher price is so popular because of the high income of the customers.
# 

# In[ ]:





# ### Usage based product distribution

# In[429]:


pd.crosstab(df['Usage'], df['Product'], margins=True)


# In[430]:


pd.crosstab(df['Usage'], df['Product'], normalize='columns')*100


# In[431]:


sns.catplot(x='Product', y='Usage', kind="box", data=df)
plt.title("Usage based product distribution")
plt.show()


# ##### Observation:
# * customers who use tredmill 6 to 7 days per week is most likely to use KP781
# * KP281 and KP481 users are the ones who use thier tredmill 2 to 4 days per week.

# In[ ]:





# ### Fitness based product distribution

# In[432]:


pd.crosstab(df['Fitness_category'], df['Product'], margins=True)


# In[433]:


pd.crosstab(df['Fitness_category'], df['Product'], normalize='columns')*100


# In[434]:


pd.crosstab(df['Fitness_category'], df['Product']).plot(kind="bar", stacked=False, rot=45)
plt.title("Education based product distribution")
plt.legend()
plt.show()


# ##### Observation:
# * if the person is in excellent shape , the probabiliy that he is using KP781 is more than 70%.
# 

# In[ ]:





# ### Gender wise Miles per use

# In[435]:


sns.catplot(x= "Product", y = "Miles per 1 use",hue="Gender" ,kind = "box", data=df, )
plt.show()


# ##### Observation:
# * Female Customers who are running average 40 miles/usage (extensive exercise) , are using product KP781, which is higher than Male average using same product.
# * KP781 can be recommended for Female customers who exercises extensively.
# * Males customers who are running average of 28 miles/usage (average exercise) , are using product KP281 .
# * Males customers who are running average of 32 miles/usage (average exercise) , are using product KP481 . and for female average running for same product is 28 miles/usage.
# 

# ### Ftiness category analysis

# In[436]:


pd.crosstab(index=[df["Product"],df["Fitness_category"]],columns=df["Gender"])


# In[ ]:





# In[437]:


df[df['Miles per 1 use']>40]['Fitness_category'].value_counts()


# In[438]:


df[df['Miles per 1 use']>np.percentile(df['Miles per 1 use'],95)]['Fitness_category'].value_counts()


# In[439]:


pd.crosstab([df["Product"],df["Gender"]],df["Fitness"],margins=True)


# In[ ]:





# ##### Observation:
# * Mejority of customers who are in Excellent shape, uses KP781
# * Mejority of customers who are in Average shape, uses KP281
# * combined fitness level of Male and Female customers who uses KP281 is equal.
# * Mles customers who uses KP781 is the second highest group of people in fitness category.

# In[ ]:





# ### Miles based Product distribution

# In[473]:


sns.catplot(x= "Fitness_category", y = "Miles" ,kind = "box",hue="Product", data= df.sort_values(by="Fitness"))
plt.xticks(rotation = 90)
plt.title("Miles based Product Distribution",fontdict ={"fontsize": 17})
plt.show()


# ##### Observation:
# * People who run/walk more miles(>130) , are more likely to use KP781 product.
# * People who walk/run around 60 to 130 miles are more likely to use KP281 and KP481 products.

# In[ ]:





# ### correlations between all features

# In[474]:


df.corr()


# In[475]:


plt.figure(figsize=(16,9))
sns.heatmap(df.corr(),annot=True)
plt.show()


# #### correlation : >0.6

# In[476]:


print(df.corr().unstack()[(df.corr().unstack()>0.65) & (df.corr().unstack()!=1)])


# ##### Observation:
# 
# * Fitness and Miles have a positive and a very high correlation: 0.79
# * Product price and Income have a positive and a very high correlation: 0.7

# In[ ]:





# In[477]:


df.head()


# In[478]:


d = df[["Age","Education","Fitness","Income","Miles","Gender"]]

x = sns.pairplot(d, kind = "reg", hue="Gender")
x.map_diag(sns.kdeplot)
plt.show()


# In[ ]:





# In[ ]:





# """
# ### From observations 1 to 12, we can make a customer profile for particular products:
# 
# #### KP281 : 
# * Most affordable and entry level and Maximum Selling Product. 
# * This model popular amongst both Male and Female customers 
# * Same number of Male and Female customers. 
# * Customers walk/run average 70 to 90 miles on this product. 
# * Customers use 3 to 4 times a week 
# * Fitness Level of this product users is Average Shape. 
# * More general purpose for all age group and fitness levels.
# #### KP481 : 
# * Intermediate Price Range 
# * Fitness Level of this product users varies from Bad to Average Shape depending on their usage. 
# * Customers prefer KP481 model to use less frequent but to run more miles per week on this. 
# * Customer walk/run average 70 to 130 or more miles per week on his product. 
# * has higher probability of selling for female customers. 
# * Probability of Female customer buying KP481 is significantly higher than male. 
# * KP481 product is specifically recommended for Female customers who are intermediate user. 
# * customers are from adult, teen and mid-age categories.
# #### KP781 : 
# * least sold product. 
# * high price and preferred by customers who does exercises more extensively and run more miles. 
# * Customer walk/run average 120 to 200 or more miles per week on his product. 
# * Customers use 4 to 5 times a week at least. 
# * If person is in Excellent Shape , the probability that he is using KP781 is more than 90%. 
# * Female Customers who are running average 180 miles (extensive exercise) , are using product KP781, which is higher than Male average using same product. 
# * KP781 can be recommended for Female customers who exercises extensively. 
# * Probability of Male customer buying Product KP781(31.73%) is way more than female(9.21%). 
# * Probability of a single person buying KP781 is higher than Married customers. So , KP781 is also recommended for people who are single and exercises more. 
# * most of old people who are above 45 age and adult uses this product.
# 
# ## Recommendations : 
# * Recommend KP781 product to users who exercises/run more frequently and run more and more miles , and have high income. Since Kp781 is least selling product (22.2% share of all the products) , recommend this product some customers who exercise at intermediate to extensive level , if they are planning to go for KP481. Also the targeted Age Category is Adult and age above 45. 
# * Recommend KP481 product specifically for female customers who run/walk more miles , as data shows their probability is higher. Statistical Summery about fitness level and miles for KP481 is not good as KP281 which is cheaper product. Possibly because of price, customers prefer to purchase KP281. It is recommended to make some necessary changes to product K481 to increase customer experience. 
# """

# In[ ]:




