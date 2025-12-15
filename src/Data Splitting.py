# Data Splitting

# Memisahkan fitur(X) dan target(y)
target = 'Gallstone Status'
X = data.drop(columns=[target])
y = data[target]

# Split data training dan testing (training 80%, testing 20%)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Jumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))