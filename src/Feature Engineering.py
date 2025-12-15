# Feature Engineering

# Menambahkan rasio kolesterol (TC/HDL dan LDL/HDL) dan AIP
eps = np.finfo(float).eps
data['TC_to_HDL'] = data['Total Cholesterol (TC)'] / data['High Density Lipoprotein (HDL)'].replace(0, eps)
data['LDL_to_HDL'] = data['Low Density Lipoprotein (LDL)'] / data['High Density Lipoprotein (HDL)'].replace(0, eps)
data['AIP'] = np.log(data['Triglyceride'] / data['High Density Lipoprotein (HDL)'].replace(0, eps))