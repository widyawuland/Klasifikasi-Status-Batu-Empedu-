# Data Cleaning

#cek informasi dari dataset Gallstone
data.info()

data.head()

#cek nilai yang hilang/missing values
missing_values=data.isnull().sum()

if missing_values.any():
    print("Ada missing values", missing_values)
else:
    print("Tidak ada missing values")

#cek duplikat data yang ada didalam dataset
num_duplicates = data.duplicated().sum()
if num_duplicates > 0:
    print(f"\nMenghapus {num_duplicates} data duplikat.")
    data.drop_duplicates(inplace=True)
else:
    print("\nTidak ditemukan data duplikat.")

#cek tipedata
data.dtypes

#cek outliers pada dataset
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

outlier_mask = (data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))

outlier_count = outlier_mask.sum()
print("jumlah outliers pada kolom")
print(outlier_count)

# cek imbalance data

import matplotlib.pyplot as plt
import seaborn as sns

#hitung jumlah tiap kelas
class_counts = data['Gallstone Status'].value_counts()

# Hitung persentase tiap kelas
class_percent = data['Gallstone Status'].value_counts(normalize=True) * 100

# Tampilkan jumlah & persentase
imbalance_df = pd.DataFrame({
    'Jumlah Data': class_counts,
    'Persentase (%)': class_percent.round(2)
})

print(imbalance_df)

# Visualisasi distribusi kelas
plt.figure(figsize=(5,4))
sns.countplot(x='Gallstone Status', data=data)
plt.title('Distribusi Kelas Gallstone Status')
plt.xlabel('Gallstone Status')
plt.ylabel('Jumlah Data')
plt.show()

#cek noise
# skala ECF/TBW (ubah persen → rasio)
if "Extracellular Fluid/Total Body Water (ECF/TBW)" in data.columns:
    data["Extracellular Fluid/Total Body Water (ECF/TBW)"] = \
        data["Extracellular Fluid/Total Body Water (ECF/TBW)"] / 100

# batas wajar: 0 – 200
corrected_VMA_limit = (0, 200)

noise_limits = {
    "Age": (0, 120),
    "Height": (80, 250),  # cm
    "Weight": (20, 300),  # kg
    "Body Mass Index (BMI)": (5, 80),
    "Total Body Water (TBW)": (10, 80),
    "Extracellular Water (ECW)": (5, 40),
    "Intracellular Water (ICW)": (5, 60),
    "Extracellular Fluid/Total Body Water (ECF/TBW)": (0.3, 0.5),
    "Total Body Fat Ratio (TBFR) (%)": (1, 80),
    "Lean Mass (LM) (%)": (10, 90),
    "Body Protein Content (Protein) (%)": (5, 40),
    "Visceral Fat Rating (VFR)": (1, 30),
    "Bone Mass (BM)": (0.5, 10),
    "Muscle Mass (MM)": (5, 150),
    "Obesity (%)": (1, 100),
    "Total Fat Content (TFC)": (1, 100),
    "Visceral Fat Area (VFA)": (1, 300),
    "Visceral Muscle Area (VMA) (Kg)": corrected_VMA_limit,
    "Glucose": (40, 500),
    "Total Cholesterol (TC)": (50, 500),
    "Low Density Lipoprotein (LDL)": (10, 400),
    "High Density Lipoprotein (HDL)": (5, 200),
    "Triglyceride": (20, 1500),
    "Aspartat Aminotransferaz (AST)": (5, 500),
    "Alanin Aminotransferaz (ALT)": (5, 500),
    "Alkaline Phosphatase (ALP)": (10, 1000),
    "Creatinine": (0.2, 10),
    "Glomerular Filtration Rate (GFR)": (10, 200),
    "C-Reactive Protein (CRP)": (0, 300),
    "Hemoglobin (HGB)": (5, 25),
    "Vitamin D": (1, 150)
}

noise_report = {}

for col in noise_limits:
    low, high = noise_limits[col]
    if col in data.columns:
        count_noise = data[(data[col] < low) | (data[col] > high)].shape[0]
        noise_report[col] = count_noise
# Tampilkan hasil
pd.DataFrame.from_dict(noise_report, orient='index', columns=["Jumlah Noise"])


cols_check = [
    "Extracellular Fluid/Total Body Water (ECF/TBW)",
    "Visceral Muscle Area (VMA) (Kg)"
]

data[cols_check].describe()