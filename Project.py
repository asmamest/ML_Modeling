import pandas as pd # Sataset Manipulaton
import matplotlib.pyplot as plt # Graphics Creation
import seaborn as sns #Advanced Statistics Visualisation 
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


DataFrame = pd.read_csv(r'C:\\Users\\user\\Desktop\\NV Projet\\archive\\predictive_maintenance.csv') 
print(DataFrame.columns)

# Visualizing individual variables
# Air temperature [K] : 
# sns.histplot(DataFrame['Air temperature [K]'],kde=True)
# plt.title("Air temperature Distribution [K]")
# plt.show()

# # Process temperature [K]
# sns.histplot(DataFrame['Process temperature [K]'], kde=True)
# plt.title("Process temperature Distribution  [K]")
# plt.show()

# # Rotational speed

# sns.histplot(DataFrame['Rotational speed [rpm]'],kde=True)
# plt.title("Rotational speed Distribution [rpm]")
# plt.show()

# # Torque 

# sns.histplot(DataFrame['Torque [Nm]'],kde=True)
# plt.title("Torque Distribution [Nm]")
# plt.show()

# #Tool wear 

# sns.histplot(DataFrame['Tool wear [min]'],kde=True)
# plt.title('Tool wear Distribution [min]')
# plt.show()


# #  Visualizing categorical variables
# # Target 

# sns.countplot(x='Target', data=DataFrame)
# plt.title("Target (Failure or No) Repartition")
# plt.show()

# # Failure Type 
# sns.countplot(x='Failure Type', data=DataFrame)
# plt.title("Failure Type Repartition")
# plt.show()


# Count the frequencies of each category

# countAirT = DataFrame['Air temperature [K]'].value_counts()
# # print(countAirT)
# countProcessT = DataFrame['Process temperature [K]'].value_counts()
# # print(countProcessT)
# countRSpeed = DataFrame['Rotational speed [rpm]'].value_counts()
# # print(countRSpeed)
# countTorque = DataFrame['Torque [Nm]'].value_counts()
# # print(countTorque)
# countToolwear  = DataFrame['Tool wear [min]'].value_counts()
# # print(countToolwear)
# countTarget = DataFrame['Target'].value_counts()
# # print(countTarget)
# countFailureType = DataFrame['Failure Type'].value_counts()
# # print(countFailureType)
# # Data Exploration 
# # print(DataFrame.head())
# # print(DataFrame.info())

# # Analyze the Distribution 

# PourcentageAirT = (countAirT / len(DataFrame['Air temperature [K]']))*100
# print(PourcentageAirT)
# PourcentageProcessT = (countProcessT / len(DataFrame['Process temperature [K]']))*100
# print(PourcentageProcessT)
# PourcentageRSpeed = (countRSpeed / len(DataFrame['Rotational speed [rpm]']))*100
# print(PourcentageRSpeed)
# PourcentageTorque = (countTorque / len(DataFrame['Torque [Nm]']))*100
# print(PourcentageTorque)
# PourcentageToolwear = (countToolwear / len(DataFrame['Tool wear [min]']))*100
# print(PourcentageToolwear)
# PourcentageTarget = (countTarget / len(DataFrame['Target']))*100
# print(PourcentageTarget)
# PourcentageFailureType = (countFailureType / len(DataFrame['Failure Type']))*100
# print(PourcentageFailureType)

# Cleaning : 

DataFrame = DataFrame.drop(columns=['UDI', 'Product ID'])


# Classes Separation : Failure Type

df_majority = DataFrame[DataFrame['Failure Type'] == 'No Failure']
df_minority = DataFrame[DataFrame['Failure Type'] != 'No Failure']

# Down_sampling the majority class

df_majority_downsampled = resample(df_majority,
                                   replace = False ,
                                   n_samples = len(df_minority),
                                   random_state = 42)

# Class combination
df_balanced = pd.concat([df_majority_downsampled,df_minority])

# Data shuffling (to be added)

df_balanced = df_balanced.sample(frac=1,random_state=42).reset_index(drop=True)

# Select numeric columns
cols_to_scale = ['Air temperature [K]', 'Process temperature [K]', 
                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

scaler = StandardScaler()
DataFrame[cols_to_scale ] = scaler.fit_transform(DataFrame[cols_to_scale ])

# save a modified DataFrame

df_balanced.to_csv(r'C:\\Users\\user\\Desktop\\NV Projet\\archive\\predictive_maintenance_balanced.csv')

# Encoding of categorical variables

# Encoding of "Failure Type"

le = LabelEncoder()

df_balanced['Failure Type'] = le.fit_transform(df_balanced['Failure Type'])

# Encoding of "Type"

df_balanced['Type'] = le.fit_transform(df_balanced['Type'])

print(df_balanced.head())


# Visualizing individual variables
# Air temperature [K] : 
sns.histplot(df_balanced['Air temperature [K]'],kde=True)
plt.title("Air temperature Distribution [K]")
plt.show()

# Process temperature [K]
sns.histplot(df_balanced['Process temperature [K]'], kde=True)
plt.title("Process temperature Distribution  [K]")
plt.show()

# Rotational speed

sns.histplot(df_balanced['Rotational speed [rpm]'],kde=True)
plt.title("Rotational speed Distribution [rpm]")
plt.show()

# Torque 

sns.histplot(df_balanced['Torque [Nm]'],kde=True)
plt.title("Torque Distribution [Nm]")
plt.show()

#Tool wear 

sns.histplot(df_balanced['Tool wear [min]'],kde=True)
plt.title('Tool wear Distribution [min]')
plt.show()


#  Visualizing categorical variables
# Target 

sns.countplot(x='Target', data=df_balanced)
plt.title("Target (Failure or No) Repartition")
plt.show()

# Failure Type 
sns.countplot(x='Failure Type', data=df_balanced)
plt.title("Failure Type Repartition")
plt.show()