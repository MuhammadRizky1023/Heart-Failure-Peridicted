## Laporan Proyek Heart Disease Prediction - Muhammad Rizky
Domain Proyek

Penyakit jantung merupakan salah satu penyebab utama kematian di dunia. Dengan adanya perkembangan teknologi dan data medis yang besar, machine learning dapat digunakan untuk membantu memprediksi risiko seseorang terkena penyakit jantung berdasarkan faktor-faktor seperti usia, jenis kelamin, tekanan darah, kadar kolesterol, dan lainnya.

Pentingnya prediksi ini adalah untuk membantu klinik dan rumah sakit dalam memberikan perawatan lebih awal dan mencegah risiko fatal bagi pasien.

## Business Understanding
Problem Statements
Bagaimana cara memprediksi penyakit jantung dengan menggunakan data medis seperti usia, tekanan darah, dan kadar kolesterol?
Algoritma apa yang memberikan hasil prediksi terbaik untuk kasus penyakit jantung?

## Goals
 Membangun model machine learning yang mampu memprediksi apakah seseorang memiliki risiko terkena penyakit jantung atau tidak.
 Membandingkan performa beberapa algoritma seperti K-Nearest Neighbor (KNN), Random Forest, dan Boosting, untuk menemukan model terbaik.

Solution Statements
Kami akan menggunakan tiga algoritma: K-Nearest Neighbor (KNN), Random Forest, dan Boosting untuk memecahkan masalah prediksi.
Setelah membangun model dasar, hyperparameter tuning dilakukan untuk meningkatkan performa model, terutama pada algoritma Random Forest dan Boosting.

## Data Understanding


Dataset yang digunakan pada proyek ini dibuat oleh [fedesoriano](https://www.kaggle.com/fedesoriano), yang diunggah ke [Kaggle](https://www.kaggle.com/) pada Desember 2021. Dataset ini dapat ditemukan di [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

Dataset ini terdiri dari **918 baris** dan **12 kolom**, yang mencakup variabel-variabel klinis pasien, termasuk usia, tekanan darah, kolesterol, dan kondisi jantung pasien yang direpresentasikan dalam variabel target **HeartDisease**.

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


## Exploratory Data Analysis (EDA)

Exploratory data analysis merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.
### Distribusi Variabel Kategori

Distribusi dari beberapa variabel kategori dalam dataset seperti **Jenis Kelamin (Sex)**, **Tipe Nyeri Dada (ChestPainType)**, dan **Hasil EKG Istirahat (RestingECG)** ditampilkan untuk melihat sebaran data.

![Distribusi Variabel Kategori](https://drive.google.com/uc?export=view&id=1iWavfllfj_mdKjjgGSLPZV1Zo9x3nhW7)

### Distribusi Variabelr Numerik

Distribusi dari beberapa variabel kategori dalam dataset seperti **Umur (Age)**, **Tekanan Darah Istirahat (RestingBP)**, dan **Kolesterol (Cholesterol)** ditampilkan untuk melihat sebaran data.

![Korelasi Fitur Numerik](https://drive.google.com/uc?export=view&id=15HJoVQTBp3p7pfnm-ciBOL26bfChBvxn)

## Exploratory Data Analysis (EDA) Univariate Analysis
Univariate analysis dilakukan untuk memahami distribusi dari setiap fitur secara individual. Kita akan memeriksa apakah data terdistribusi normal, apakah ada skewness, dan bagaimana distribusi nilai dalam setiap variabel.

## Exploratory Data Analysis (EDA) Multivariate Analysis
### Korelasi Antar Fitur Numerik

Korelasi antara fitur numerik seperti **Umur (Age)**, **Tekanan Darah Istirahat (RestingBP)**, dan **Kolesterol (Cholesterol)** menunjukkan hubungan antar variabel. Gambar heatmap korelasi berikut menggambarkan kekuatan hubungan antara fitur-fitur ini.

![Korelasi Fitur Numerik](https://drive.google.com/uc?export=view&id=1qyKaitq5zHg2oRnILUXfJmjpgV7KWukh)


## Data Preparation

Setelah memahami data, kami melakukan beberapa tahapan Data Preparation untuk mempersiapkan data sebelum model diimplementasikan. Langkah-langkah ini termasuk:

    Handling Missing Values Berdasarkan hasil eksplorasi, tidak ada missing values dalam dataset ini sehingga tidak diperlukan proses imputasi atau penghapusan data.

    Encoding Fitur Kategori Untuk fitur kategori seperti Sex, ChestPainType, RestingECG, ExerciseAngina, dan ST_Slope, kami melakukan encoding agar data bisa digunakan oleh model machine learning. Kami menggunakan One-Hot Encoding untuk fitur dengan lebih dari dua kategori.


df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

Feature Scaling Kami menggunakan StandardScaler untuk menormalisasi fitur numerik seperti Age, RestingBP, Cholesterol, MaxHR, dan Oldpeak agar memiliki skala yang sama dan memudahkan algoritma untuk bekerja lebih baik.


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])

Train-Test Split Kami membagi dataset menjadi data latih (80%) dan data uji (20%) untuk memastikan evaluasi model dilakukan pada data yang belum pernah dilihat model.

    from sklearn.model_selection import train_test_split
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



## Model Development

Pada bagian ini, tiga model telah dibangun dan diuji:

K-Nearest Neighbor (KNN): Model sederhana yang bekerja dengan mengukur jarak antara sampel dan menentukan kelas berdasarkan mayoritas tetangga terdekat.
Random Forest (RF): Model ensemble yang membangun banyak pohon keputusan dan mengambil hasil berdasarkan voting dari tiap pohon.
Boosting: Model yang berfokus pada memperbaiki prediksi yang salah dari model sebelumnya.

Improvement: Hyperparameter tuning dilakukan pada Random Forest dan Boosting untuk memaksimalkan akurasi model dengan menggunakan grid search. Setelah tuning, Random Forest menunjukkan hasil yang paling baik dengan performa yang stabil.
Evaluation

Model dievaluasi menggunakan metrik Mean Squared Error (MSE) dan akurasi. Berikut hasil dari setiap model:

   
| Model                   | MSE Latih  | MSE Uji    | Akurasi |
|-------------------------|------------|------------|---------|
| K-Nearest Neighbor (KNN) | 0.00012    | 0.00037    | 78%     |
| Random Forest (RF)       | 0.0        | 0.000158   | 84%     |
| Boosting                 | 0.000061   | 0.000141   | 82%     |


## Improvement Model dengan HyperTurning Hyperparameter Tuning
Proses hyperparameter tuning dilakukan untuk Random Forest dan Boosting menggunakan GridSearchCV. Tuning ini meningkatkan akurasi dan menurunkan MSE, khususnya pada Random Forest yang tetap menjadi model terbaik.

## Grafik Peforma Model
### Visualisasi MSE untuk Model yang Berbeda

Pada grafik di bawah ini, kita dapat melihat perbandingan **MSE** dari beberapa model yang digunakan. Grafik ini memvisualisasikan MSE baik pada data latih maupun data uji untuk setiap model.

![MSE Model](https://drive.google.com/uc?export=view&id=1AtOWiV-bxgRogeZLHYB2wquHX-RjKzdD)
- **RF_turned** dan **Boosting** memiliki performa yang baik pada data latih dan uji, dengan nilai MSE uji yang rendah.
- **KNN** menunjukkan perbedaan yang signifikan antara MSE latih dan uji, yang mungkin menunjukkan **overfitting**.
- **Random Forest (RF)** pada versi tuning maupun tidak menunjukkan bahwa performanya tetap baik dengan MSE uji yang rendah.

Dari evaluasi ini, dapat disimpulkan bahwa model **Random Forest (RF)** dan **Boosting** memberikan performa yang lebih baik dibandingkan model lainnya dalam hal MSE dan akurasi

## Referensi:

Rahman, S. et al. (2022). "Predicting Heart Disease Using Machine Learning Techniques". Journal of Healthcare Engineering.
World Health Organization (2021). "Cardiovascular diseases (CVDs)".
aleema S., S. Syed dan M. Ahmad, "Application of Machine Learning Algorithms for Heart Disease Prediction," International Journal of Advanced Computer Science and Applications, vol. 11, no. 4, 2020.
