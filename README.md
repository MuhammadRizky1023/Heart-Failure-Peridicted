## Laporan Proyek Heart Disease Prediction - Muhammad Rizky
## Domain Proyek
Penyakit jantung merupakan salah satu penyebab utama kematian di dunia. Dengan adanya perkembangan teknologi dan data medis yang besar, machine learning dapat digunakan untuk membantu memprediksi risiko seseorang terkena penyakit jantung berdasarkan faktor-faktor klinis seperti usia, jenis kelamin, tekanan darah, kadar kolesterol, dan lainnya. Prediksi ini sangat penting untuk mendukung upaya pencegahan dini serta membantu klinik dan rumah sakit memberikan perawatan lebih awal kepada pasien berisiko tinggi (Rahman et al., 2022; WHO, 2021).


## Business Understanding
### Problem Statements
- Bagaimana cara memprediksi risiko penyakit jantung berdasarkan data medis?
- Algoritma machine learning apa yang memberikan performa terbaik dalam memprediksi penyakit jantung?

### Goals
 - Membangun model machine learning yang dapat memprediksi apakah seseorang berisiko terkena penyakit jantung.
 - Membandingkan performa beberapa algoritma seperti K-Nearest Neighbor (KNN), Random Forest, dan Boosting untuk menemukan model terbaik.

   ### Solution Statements
   - Menggunakan tiga algoritma utama: K-Nearest Neighbor (KNN), Random Forest, dan Boosting.
   - Melakukan hyperparameter tuning pada Random Forest dan Boosting untuk meningkatkan akurasi model.

## Data Understanding


Dataset yang digunakan pada proyek ini dibuat oleh [fedesoriano](https://www.kaggle.com/fedesoriano), yang diunggah ke [Kaggle](https://www.kaggle.com/) pada Desember 2021. Dataset ini dapat ditemukan di [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

Dataset ini terdiri dari **918 baris** dan **12 kolom**, yang mencakup variabel-variabel klinis pasien, termasuk usia, tekanan darah, kolesterol, dan kondisi jantung pasien yang direpresentasikan dalam variabel target **HeartDisease**.
### Variabel-variabel pada Heart Failure dataset adalah sebagai berikut:

| **Fitur**          | **Deskripsi**                                                                         |
|--------------------|---------------------------------------------------------------------------------------|
| **Age**            | Umur pasien (dalam tahun)                                                             |
| **Sex**            | Jenis kelamin pasien (M = Pria, F = Wanita)                                            |
| **ChestPainType**   | Tipe nyeri dada (TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic) |
| **RestingBP**      | Tekanan darah saat istirahat (mm Hg)                                                  |
| **Cholesterol**    | Kadar kolesterol serum dalam darah (mm/dl)                                             |
| **FastingBS**      | Gula darah puasa (1 = jika FastingBS > 120 mg/dl, 0 = jika FastingBS <= 120 mg/dl)     |
| **RestingECG**     | Hasil elektrokardiografi istirahat (Normal, ST, LVH)                                   |
| **MaxHR**          | Detak jantung maksimal yang dicapai                                                    |
| **ExerciseAngina** | Angina akibat aktivitas fisik (Y = Yes, N = No)                                        |
| **Oldpeak**        | Depresi ST yang disebabkan oleh latihan relatif terhadap istirahat                     |
| **ST_Slope**       | Kemiringan segmen ST saat puncak latihan (Up, Flat, Down)                              |
| **HeartDisease**   | Klasifikasi penyakit jantung (1 = memiliki penyakit jantung, 0 = tidak)                |

### Kondisi Data
- Missing Values: Tidak ada data kosong dalam dataset ini.
- Duplikat: Tidak ditemukan data duplikat dalam dataset.


## Exploratory Data Analysis (EDA)

Exploratory data analysis merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

### Exploratory Data Analysis (EDA) Univariate Analysis
Univariate analysis dilakukan untuk memahami distribusi dari setiap fitur secara individual. Kita akan memeriksa apakah data terdistribusi normal, apakah ada skewness, dan bagaimana distribusi nilai dalam setiap variabel.
### Distribusi Variabel Kategori

Distribusi dari beberapa variabel kategori dalam dataset seperti **Jenis Kelamin (Sex)**, **Tipe Nyeri Dada (ChestPainType)**, dan **Hasil EKG Istirahat (RestingECG)** ditampilkan untuk melihat sebaran data.

![Distribusi Variabel Kategori](https://drive.google.com/uc?export=view&id=1iWavfllfj_mdKjjgGSLPZV1Zo9x3nhW7)

### Distribusi Variabelr Numerik

Distribusi dari beberapa variabel kategori dalam dataset seperti **Umur (Age)**, **Tekanan Darah Istirahat (RestingBP)**, dan **Kolesterol (Cholesterol)** ditampilkan untuk melihat sebaran data.

![Korelasi Fitur Numerik](https://drive.google.com/uc?export=view&id=15HJoVQTBp3p7pfnm-ciBOL26bfChBvxn)


## Exploratory Data Analysis (EDA) Multivariate Analysis
### Korelasi Antar Fitur Numerik

Korelasi antara fitur numerik seperti **Umur (Age)**, **Tekanan Darah Istirahat (RestingBP)**, dan **Kolesterol (Cholesterol)** menunjukkan hubungan antar variabel. Gambar heatmap korelasi berikut menggambarkan kekuatan hubungan antara fitur-fitur ini.

![Korelasi Fitur Numerik](https://drive.google.com/uc?export=view&id=1qyKaitq5zHg2oRnILUXfJmjpgV7KWukh)


## Data PreparationPada tahap ini, dilakukan beberapa proses untuk mempersiapkan data sebelum model machine learning diterapkan.
## Feature Engineering

Dilakukan One-Hot Encoding untuk fitur kategori seperti Sex, ChestPainType, RestingECG, ExerciseAngina, dan ST_Slope.


df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

## Feature Scaling

Fitur numerik seperti Age, RestingBP, Cholesterol, MaxHR, dan Oldpeak dinormalisasi menggunakan StandardScaler untuk menghindari bias pada model.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])

## Split Data

Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan fungsi train_test_split.

from sklearn.model_selection import train_test_split
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Model Development

Tiga algoritma machine learning diterapkan untuk memprediksi penyakit jantung: K-Nearest Neighbor (KNN), Random Forest, dan Boosting. Setiap algoritma dieksplorasi dengan tuning hyperparameter untuk meningkatkan performa.
K-Nearest Neighbor (KNN)

KNN bekerja dengan mencari tetangga terdekat dan mengklasifikasikan data berdasarkan mayoritas kelas tetangga tersebut. Parameter penting dalam model ini adalah jumlah tetangga k.
Random Forest

Random Forest adalah model ensemble yang terdiri dari banyak decision trees. Tiap pohon keputusan dibangun dari subset data dan hasil akhirnya didapat dari voting mayoritas tiap pohon. Beberapa hyperparameter penting termasuk jumlah pohon (n_estimators) dan kedalaman pohon (max_depth).
Boosting

Boosting adalah metode ensemble yang meningkatkan akurasi model dengan mengkombinasikan beberapa model lemah yang terus diperbaiki pada iterasi berikutnya. Algoritma yang digunakan adalah Gradient Boosting.


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

### Implementasi model KNN, Random Forest, dan Boosting
knn_model = KNeighborsClassifier(n_neighbors=5)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)



## Evaluasi

Karena ini adalah masalah klasifikasi, evaluasi model dilakukan menggunakan metrik seperti accuracy, precision, recall, dan F1 score, bukan Mean Squared Error (MSE). Berikut hasil evaluasi dari model:
### Hasil Evaluasi
| **Model**                 | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|---------------------------|--------------|---------------|------------|--------------|
| K-Nearest Neighbor (KNN)   | 78%          | 0.76          | 0.80       | 0.78         |
| Random Forest (RF)         | 84%          | 0.85          | 0.83       | 0.84         |
| Boosting                  | 82%          | 0.82          | 0.81       | 0.81         |


## Visualisasi Plot
### Visualisasi MSE untuk Model yang Berbeda

Pada grafik di bawah ini, kita dapat melihat perbandingan **MSE** dari beberapa model yang digunakan. Grafik ini memvisualisasikan MSE baik pada data latih maupun data uji untuk setiap model.

![MSE Model](https://drive.google.com/uc?export=view&id=1AtOWiV-bxgRogeZLHYB2wquHX-RjKzdD)
- **RF_turned** dan **Boosting** memiliki performa yang baik pada data latih dan uji, dengan nilai MSE uji yang rendah.
- **KNN** menunjukkan perbedaan yang signifikan antara MSE latih dan uji, yang mungkin menunjukkan **overfitting**.
- **Random Forest (RF)** pada versi tuning maupun tidak menunjukkan bahwa performanya tetap baik dengan MSE uji yang rendah.

Dari evaluasi ini, dapat disimpulkan bahwa model **Random Forest (RF)** dan **Boosting** memberikan performa yang lebih baik dibandingkan model lainnya dalam hal MSE dan akurasi

### Grafik Akurasi Model

Grafik di bawah menunjukkan perbandingan **akurasi** dari beberapa model:

![Akurasi for Different Models](https://drive.google.com/uc?export=view&id=1Lr7E1y-SV_a72z-gaNqviZ3Rq6QFIiDg)

Dari grafik akurasi tersebut, kita dapat melihat bahwa:

- **Random Forest (RF)**, terutama yang sudah dituning (RF_turned), memiliki **akurasi tertinggi**, baik pada data latih maupun data uji.
- **Boosting** juga memiliki akurasi yang sangat baik, meskipun sedikit di bawah **Random Forest**.
- **K-Nearest Neighbor (KNN)** memiliki akurasi yang lebih rendah dibandingkan model lainnya, baik pada data latih maupun data uji.


## Kesimpulan

    - Random Forest memberikan hasil terbaik dengan akurasi 84% dan stabil di berbagai metrik evaluasi.
   -  Boosting memberikan hasil yang baik tetapi sedikit di bawah Random Forest.
   -  KNN memiliki akurasi yang lebih rendah, kemungkinan disebabkan oleh overfitting pada data latih.

Dengan demikian, model Random Forest menjadi pilihan terbaik untuk prediksi penyakit jantung dalam proyek ini.

## Referensi:
- Rahman, S. et al. (2022). "Predicting Heart Disease Using Machine Learning Techniques". Journal of Healthcare Engineering.
- World Health Organization (2021). "Cardiovascular diseases (CVDs)". Available at: https://www.who.int.
- aleema S., S. Syed dan M. Ahmad, "Application of Machine Learning Algorithms for Heart Disease Prediction," International Journal of Advanced Computer Science and Applications, vol. 11, no. 4, 2020.
