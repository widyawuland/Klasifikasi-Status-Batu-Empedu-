# 📘 Judul Proyek
*Analisis Klasifikasi Status Batu Empedu pada Dataset Klinis Menggunakan Machine Learning dan Deep Learning*

## 👤 Informasi
- **Nama:** [Widya Wulandari]  
- **Repo:** [...]  
- **Video:** [...]  

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
- [Bagaimana membangun model machine learning dan deep learning untuk mengklasifikasikan status batu empedu berdasarkan data bioimpedansi dan parameter laboratorium?]
- [Model mana yang memberikan performa terbaik dalam memprediksi risiko batu empedu?]
- [Apakah model deep learning mampu memberikan peningkatan performa dibandingkan model machine learning tradisional?]

**Goals:**  
- [Mengembangkan model klasifikasi status batu empedu menggunakan data tabular klinis.]
- [Membandingkan performa baseline model, advanced machine learning model, dan deep learning model.]
- [Mengevaluasi performa model menggunakan metrik klasifikasi yang relevan.]
  

---
## 📁 Struktur Folder
```
project/
│
├── data/                   
│   └── gallstone_.csv
│ 
├── notebooks/              
│   └── ML_Project.ipynb
│
├── src/                    
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
└── Laporan Proyek Machine Learning.docx
```
---

# 3. 📊 Dataset
- **Sumber:** [https://www.kaggle.com/datasets/xixama/gallstone-dataset-uci]  
- **Jumlah Data:** [319]  
- **Tipe:** [Tabular]  

### Fitur Utama
| Fitur | Deskripsi |
| :--- | :--- |
| **Gallstone Status** | **Variabel Target:** Status keberadaan batu empedu. **0 (Ada Batu Empedu)**, **1 (Tidak Ada Batu Empedu)**. |
| **Age** | Usia orang tersebut (tahun). |
| **Gender** | Jenis kelamin orang tersebut, 0 (Laki-laki), 1 (Perempuan). |
| **Comorbidity** | Penyakit penyerta (komorbiditas): 0 (Tidak ada), 1 (Satu kondisi), 2 (Dua kondisi), 3 (Tiga atau lebih kondisi). |
| **Coronary Artery Disease (CAD)** | Penyakit kardiovaskular: 0 (Tidak), 1 (Ya). |
| **Hypothyroidism** | Kelenjar tiroid yang kurang aktif atau Hipotiroidisme: 0 (Tidak), 1 (Ya). |
| **Hyperlipidemia** | Kadar lemak tinggi dalam darah: 0 (Tidak), 1 (Ya). |
| **Diabetes Mellitus (DM)** | Gula darah tinggi: 0 (Tidak), 1 (Ya). |
| **Height** | Tinggi badan. |
| **Weight** | Berat badan. |
| **Body Mass Index (BMI)** | Rasio berat badan terhadap tinggi badan. |
| **Total Body Water (TBW)** | Total air dalam tubuh. |
| **Extracellular Water (ECW)** | Semua cairan tubuh yang berada di luar sel (plasma, cairan interstisial, dll.). |
| **Intracellular Water (ICW)** | Semua cairan yang terkandung di dalam sel-sel tubuh. |
| **Extracellular Fluid/Total Body Water (ECF/TBW)** | Rasio kandungan air ekstraseluler terhadap total air tubuh. |
| **Total Body Fat Ratio (TBFR)** | Total Lemak Tubuh (%). |
| **Lean Mass (LM)** | Massa tubuh tanpa lemak (%). |
| **Body Protein Content (Protein)** | Kandungan Protein, Lemak, Karbohidrat, Vitamin, dan Mineral (%). |
| **Visceral Fat Rating (VFR)** | Kadar lemak organ dalam. |
| **Bone Mass (BM)** | Berat tulang. |
| **Muscle Mass (MM)** | Massa otot. |
| **Obesity** | Kelebihan lemak tubuh (%). |
| **Total Fat Content (TFC)** | Total kadar lemak. |
| **Visceral Fat Area (VFA)** | Area jaringan lemak dalam. |
| **Visceral Muscle Area (VMA)** | Area otot dalam (kg). |
| **Hepatic Fat Accumulation (HFA)** | Penumpukan lemak di hati: 0 (Tidak ada) hingga 4 (Tingkat 4/sangat parah). |
| **Glucose** | Gula darah. |
| **Total Cholesterol (TC)** | Ukuran gabungan dari semua jenis kolesterol dalam darah (HDL, LDL, Trigliserida). |
| **Low Density Lipoprotein (LDL)** | Kolesterol jahat. |
| **High Density Lipoprotein (HDL)** | Kolesterol baik. |
| **Triglyceride** | Jenis lemak yang ditemukan dalam darah. |
| **Aspartat Aminotransferaz (AST)** | Jenis enzim hati. |
| **Alanin Aminotransferaz (ALT)** | Enzim yang berhubungan dengan hati. |
| **Alkaline Phosphatase (ALP)** | Jenis enzim hati dan tulang. |
| **Creatinine** | Indikator fungsi ginjal. |
| **Glomerular Filtration Rate (GFR)** | Laju filtrasi ginjal. |
| **C-Reactive Protein (CRP)** | Indikator peradangan. |
| **Hemoglobin (HGB)** | Protein dalam darah yang membawa oksigen. |
| **Vitamin D** | Vitamin esensial untuk kesehatan tulang. |

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
- **Model 1 – Baseline:** [Logistic Regression]  
- **Model 2 – Advanced ML:** [Gradient Boosting Classifier]  
- **Model 3 – Deep Learning:** [Multilayer Perceptron (MLP)]  

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

# 8. 🔮 Future Work
Data:
[✓] Mengumpulkan lebih banyak data
[✓] Menambah variasi data
[✓] Feature engineering lebih lanjut

Model:
[✓] Mencoba arsitektur DL yang lebih kompleks
[✓] Hyperparameter tuning lebih ekstensif
[✓] Ensemble methods (combining models)
[ ] Transfer learning dengan model yang lebih besar

Deployment:
[✓] Membuat API (Flask/FastAPI)
[✓] Membuat web application (Streamlit/Gradio)
[ ] Containerization dengan Docker
[ ] Deploy ke cloud (Heroku, GCP, AWS)

Optimization:
[ ] Model compression (pruning, quantization)
[✓] Improving inference speed
[✓] Reducing model size

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
