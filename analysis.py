import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv")
#checking for null values
sb.heatmap(data.isnull())
#counting for target variable
sb.countplot(x='Situation',data=data)
#info about data
data.info()

Availabilty = pd.get_dummies(data['Situation'],drop_first=True)
data.drop(['Situation'],axis=1,inplace=True)
data1 = pd.concat([data,Availabilty],axis=1)
#correlation b/w variables
sb.heatmap(data1.corr())

data1["Total_Rainfall"].plot(kind='line')
data1["Net annual groundwater availability"].plot()
data1["Total_Usage"].plot()
plt.legend(['Total_Rainfall','Net annual groundwater availability','Total_Usage'])
#for annum ground water analysis
data1.mean()
labels='Total Rainfall','Net Annual GroundWater','Total Use','Future Available','Projected demand for domestic and industrial uses upto 2025','Natural discharge during non-monsoon season'
sizes=[14.84,13.64,8.39,5.29,1.063483,1.210483]
cols = ['c','m','r','b','g','y']
plt.pie(sizes,labels=labels,colors=cols,startangle=90,shadow=True,explode=(0,0.01,0.01,0.01,0.1,0.2),autopct='%1.1f%%')

sb.pairplot(data1, x_vars='Groundwater availability for future irrigation use', y_vars='Net annual groundwater availability', kind='scatter', diag_kind='hist',size=6.0)