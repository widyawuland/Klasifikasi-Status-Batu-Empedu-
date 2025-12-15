# Exploratory Data Analysis (EDA)

import seaborn as sns
import matplotlib.pyplot as plt

# Distribusi Target
sns.countplot(x=data["Gallstone Status"])
plt.title("Distribusi Label Gallstone")
plt.show()

# Heatmap Korelasi
plt.figure(figsize=(18,12))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Heatmap Korelasi Fitur Klinis")
plt.show()

#Boxplot Obesity (%)
sns.boxplot(y="Obesity (%)", data=data)
plt.title("Boxplot Obesity (%)")
plt.show()