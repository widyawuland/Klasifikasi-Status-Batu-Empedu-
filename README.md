# 📘 *Analisis Klasifikasi Status Batu Empedu pada Dataset Klinis Menggunakan Machine Learning dan Deep Learning*

## 👤 Informasi
- **Nama:** Widya Wulandari
- **Repo:** https://github.com/widyawuland/Klasifikasi-Status-Batu-Empedu.git
- **Video:** https://youtu.be/1G7PUQp244I?si=eK_1BDEDG3ZDbner

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk mengklasifikasikan status batu empedu (Gallstone Status) menggunakan pendekatan Machine Learning dan Deep Learning berdasarkan data klinis, bioimpedansi, dan profil biokimia pasien.
Tahapan yang dilakukan meliputi:
- Eksplorasi dan persiapan data medis
- Pembangunan tiga model klasifikasi: **Baseline**, **Advanced ML**, **Deep Learning**
- Evaluasi performa model menggunakan metrik klasifikasi medis
- Penentuan model terbaik untuk prediksi risiko batu empedu 

---

# 2. 📄 Problem & Goals
**Problem Statements:** 
- Bagaimana membangun model machine learning dan deep learning untuk mengklasifikasikan status batu empedu berdasarkan data bioimpedansi dan parameter laboratorium?
- Model mana yang memberikan performa terbaik dalam memprediksi risiko batu empedu?
- Apakah model deep learning mampu memberikan peningkatan performa dibandingkan model machine learning tradisional?

**Goals:**  
- Mengembangkan model klasifikasi status batu empedu menggunakan data tabular klinis.
- Membandingkan performa baseline model, advanced machine learning model, dan deep learning model.
- Mengevaluasi performa model menggunakan metrik klasifikasi yang relevan.
  

---
## 📁 Struktur Folder
```
project/
│
├── data/                   
│   └── gallstone_.csv
│ 
├── notebooks/              
│   └── UAS_234311056_WIDYA_WULANDARI.ipynb
│
├── src/
│   └── Download dan Load Dataset.py
│   └── Exploratory Data Analysis (EDA).py
│   └── Data Cleaning.py
│   └── Feature Engineering.py
│   └── Data Splitting.py
│   └── Data Transformation.py
│   └── Modeling.py
│   
├── models/                
│   ├── logistic_regression_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── model_cnn.h5mlp_deep_learning_model.h5
│ 
├── images/
│   └── Boxplot Obesity.png
│   └── Heatmap Korelasi Fitur Klinis.png
│   └── Distribusi Label Gallstone.png
│   └── Distribusi Kelas Gallstone Status.png
│   └── Training & Validation Accuracy per epoch.png
│   └── Training & Validation Loss per epoch.png
│   └── Confusion Matrix Logistic Regression.png
│   └── Confusion Matrix Gradient Boosting.png
│   └── Feature Importance.png
│   └── Confusion Matrix Deep Learning MLP.png
│   └── visualisasi perbandingan model.png
│ 
└── README.md
└── Cheklist Submit.md
└── LICENSE.txt
└── Laporan Proyek Machine Learning.pdf
└── requirements.txt
```
---

# 3. 📊 Dataset
- **Sumber:** https://www.kaggle.com/datasets/xixama/gallstone-dataset-uci 
- **Jumlah Data:** 319  
- **Tipe:** Tabular 

## Fitur Utama

| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
| :--- | :--- | :--- | :--- |
| **Gallstone Status** | Integer | **Variabel Target:** Status keberadaan batu empedu. **0 (Ada Batu Empedu)** dan **1 (Tidak Ada Batu Empedu)**. | 0, 1 |
| **Age** | Integer | Usia orang tersebut (tahun). | 50, 47, 61 |
| **Gender** | Categorical | Jenis kelamin orang tersebut: **0 (Laki-laki)**, **1 (Perempuan)**. | 0, 1 |
| **Comorbidity** | Categorical | Penyakit penyerta (komorbiditas): 0 (Tidak ada), 1 (Satu), 2 (Dua), 3 (Tiga atau lebih). | 0, 1, 2, 3 |
| **Coronary Artery Disease (CAD)** | Binary | Penyakit kardiovaskular: 0 (Tidak), 1 (Ya). | 0, 1 |
| **Hypothyroidism** | Binary | Kelenjar tiroid yang kurang aktif atau Hipotiroidisme: 0 (Tidak), 1 (Ya). | 0, 1 |
| **Hyperlipidemia** | Binary | Kadar lemak tinggi dalam darah: 0 (Tidak), 1 (Ya). | 0, 1 |
| **Diabetes Mellitus (DM)** | Binary | Gula darah tinggi: 0 (Tidak), 1 (Ya). | 0, 1 |
| **Height** | Integer | Tinggi badan. | 185, 176, 171 |
| **Weight** | Float | Berat badan. | 92.8, 94.5, 91.1 |
| **Body Mass Index (BMI)** | Float | Rasio berat badan terhadap tinggi badan. | 27.1, 30.5, 31.2 |
| **Total Body Water (TBW)** | Float | Total air dalam tubuh. | 52.9, 43.1, 47.2 |
| **Extracellular Water (ECW)** | Float | Semua cairan tubuh yang berada di luar sel (plasma, cairan interstisial, dll.). | 21.2, 19.5, 20.1 |
| **Intracellular Water (ICW)** | Float | Semua cairan yang terkandung di dalam sel-sel tubuh. | 31.7, 23.6, 27.1 |
| **Extracellular Fluid/Total Body Water (ECF/TBW)** | Float | Rasio kandungan air ekstraseluler terhadap total air tubuh. | 40, 45, 43 |
| **Total Body Fat Ratio (TBFR)** | Float | Total Lemak Tubuh (%). | 19.2, 32.8, 27.3 |
| **Lean Mass (LM)** | Float | Massa tubuh tanpa lemak (%). | 80.84, 67.2, 72.67 |
| **Body Protein Content (Protein)** | Float | Kandungan Protein, Lemak, Karbohidrat, Vitamin, dan Mineral (%). | 18.88, 16.68, 16.35 |
| **Visceral Fat Rating (VFR)** | Integer | Kadar lemak organ dalam. | 9, 15, 6 |
| **Bone Mass (BM)** | Float | Berat tulang. | 3.7, 3.2, 3.3 |
| **Muscle Mass (MM)** | Float | Massa otot. | 71.4, 60.3, 62.9 |
| **Obesity** | Float | Kelebihan lemak tubuh (%). | 23.4, 38.8, 41.7 |
| **Total Fat Content (TFC)** | Float | Total kadar lemak. | 17.8, 31, 24.9 |
| **Visceral Fat Area (VFA)** | Float | Area jaringan lemak dalam. | 10.6, 18.4, 16.2 |
| **Visceral Muscle Area (VMA)** | Float | Area otot dalam (kg). | 39.7, 32.7, 34 |
| **Hepatic Fat Accumulation (HFA)** | Categorical | Penumpukan lemak di hati: 0 (Tidak ada) hingga 4 (Sangat parah). | 0, 1, 2, 3, 4 |
| **Glucose** | Float | Gula darah. | 102, 94, 103 |
| **Total Cholesterol (TC)** | Float | Ukuran gabungan dari semua jenis kolesterol dalam darah. | 250, 172, 179 |
| **Low Density Lipoprotein (LDL)** | Float | Kolesterol jahat. | 175, 108, 124 |
| **High Density Lipoprotein (HDL)** | Float | Kolesterol baik. | 40, 43, 59 |
| **Triglyceride** | Float | Jenis lemak yang ditemukan dalam darah. | 134, 103, 69 |
| **Aspartat Aminotransferaz (AST)** | Float | Jenis enzim hati. | 20, 14, 18 |
| **Alanin Aminotransferaz (ALT)** | Float | Enzim yang berhubungan dengan hati. | 22, 13, 14 |
| **Alkaline Phosphatase (ALP)** | Float | Jenis enzim hati dan tulang. | 87, 46, 66 |
| **Creatinine** | Float | Indikator fungsi ginjal. | 0.82, 0.87, 1.25 |
| **Glomerular Filtration Rate (GFR)** | Float | Laju filtrasi ginjal. | 112.47, 107.1, 65.51 |
| **C-Reactive Protein (CRP)** | Float | Indikator peradangan. | 0, 0.11, 1.57 |
| **Hemoglobin (HGB)** | Float | Protein dalam darah yang membawa oksigen. | 16, 14.4, 16.2 |
| **Vitamin D** | Float | Vitamin esensial untuk kesehatan tulang. | 33, 25, 30.2 |

---

# 4. 🔧 Data Preparation

### Data Cleaning
- Tidak ditemukan missing values dan data duplikat.  
- Outliers terdeteksi pada beberapa fitur, namun **tidak dihapus** karena masih berada dalam rentang variasi biologis yang wajar.  
- Dataset tidak mengalami class imbalance.  

### Feature Engineering
Penambahan fitur klinis turunan:
- Rasio Total Cholesterol / HDL (TC/HDL)  
- Rasio LDL / HDL (LDL/HDL)  
- Atherogenic Index of Plasma (AIP)  

### Data Transformation
- Standardisasi fitur numerik menggunakan **StandardScaler**.  
- Preprocessing diimplementasikan menggunakan **Pipeline** dan **ColumnTransformer** untuk mencegah data leakage.  

### Data Splitting
- Training set: 80% (255 data)  
- Test set: 20% (64 data)  
- Metode: **Stratified Split**  
- Random state: 42   

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Logistic Regression
- **Model 2 – Advanced ML:** Gradient Boosting Classifier  
- **Model 3 – Deep Learning:** Multilayer Perceptron (MLP)  

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy

### Hasil Singkat
| Model | Score (Accuracy) | Catatan |
|-------|------------------|---------|
| Baseline (Logistic Regression) | **81.25%** | Performa terbaik, stabil, interpretabilitas tinggi |
| Advanced (Gradient Boosting) | 76.56% | Cukup baik, namun tidak mengungguli baseline |
| Deep Learning (MLP – 25 Epoch) | 73.44% | Cenderung overfitting ringan, data terbatas |

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** Logistic Regression  
- **Alasan:** Model ini menghasilkan performa paling stabil dengan nilai accuracy, F1-score, dan ROC-AUC tertinggi dibandingkan model lainnya, serta memiliki waktu pelatihan yang sangat cepat dan interpretabilitas yang baik untuk konteks medis.  
- **Insight penting:** Pada dataset klinis berukuran kecil dan bersifat tabular, model machine learning sederhana yang terkalibrasi dengan baik dapat mengungguli model yang lebih kompleks seperti deep learning. Model deep learning membutuhkan jumlah data yang lebih besar agar keunggulannya dapat terlihat secara optimal.

---

## 8. 🔮 Future Work

Berikut adalah rencana pengembangan di masa depan berdasarkan proyek ini:

### Data
- [x] Mengumpulkan lebih banyak data
- [x] Menambah variasi data
- [x] Feature engineering lebih lanjut

### Model
- [x] Mencoba arsitektur DL yang lebih kompleks
- [x] Hyperparameter tuning lebih ekstensif
- [x] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

### Deployment
- [x] Membuat API (Flask/FastAPI)
- [x] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

### Optimization
- [ ] Model compression (pruning, quantization)
- [x] Improving inference speed
- [x] Reducing model size

---

# 9. 🔁 Reproducibility
Gunakan environment:

## Environment
- **Python:** 3.12  
- **Operating System:** Windows / Linux / macOS  

## Dependencies
```txt
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tensorflow==2.14.0
```
---

# 🚀 10. Cara Menjalankan Proyek

Panduan berikut menjelaskan cara menjalankan proyek klasifikasi status batu empedu baik secara **lokal** maupun menggunakan **Google Colab**.

---

## Clone Repository

```bash
git clone https://github.com/widyawuland/Klasifikasi-Status-Batu-Empedu.git
cd Klasifikasi-Status-Batu-Empedu
```

---

## Create Virtual Environment (Opsional)

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download Dataset

Unduh dataset dari Kaggle:  
https://www.kaggle.com/datasets/xixama/gallstone-dataset-uci

Simpan sebagai:
```bash
data/gallstone_.csv
```

---

## Running the Project

### Option 1: Script Modular

```bash
python src/Download dan Load Dataset.py
python src/Exploratory Data Analysis (EDA).py
python src/Data Cleaning.py
python src/Feature Engineering.py
python src/Data Splitting.py
python src/Data Transformation.py
python src/Modeling.py
```

Output model:
```bash
models/
```

---

### Option 2: Jupyter Notebook

```bash
jupyter notebook
```

Buka:
```bash
notebooks/UAS_234311056_WIDYA_WULANDARI.ipynb
```

---

### Option 3: Google Colab

1. Buka https://colab.research.google.com
2. Upload notebook `UAS_234311056_WIDYA_WULANDARI.ipynb`
4. Run All

Estimasi waktu: **10–15 menit**
