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

![Distribusi Variabel Categori](https://github.com/user-attachments/assets/171871a4-c365-45f7-84b0-9ea1f70da292)


### Distribusi Variabel Numerik

Distribusi dari beberapa variabel kategori dalam dataset seperti **Umur (Age)**, **Tekanan Darah Istirahat (RestingBP)**, dan **Kolesterol (Cholesterol)** ditampilkan untuk melihat sebaran data.
![Distribusi Variabel numerik](https://github.com/user-attachments/assets/ecc621c6-74af-4a9b-adf5-a3a4fbad88fb)



## Exploratory Data Analysis (EDA) Multivariate Analysis
### Korelasi Antar Fitur Numerik

Korelasi antara fitur numerik seperti **Umur (Age)**, **Tekanan Darah Istirahat (RestingBP)**, dan **Kolesterol (Cholesterol)** menunjukkan hubungan antar variabel. Gambar heatmap korelasi berikut menggambarkan kekuatan hubungan antara fitur-fitur ini.

![Korelasi Fitur Numerik](https://github.com/user-attachments/assets/e231a870-2c40-436e-9b8d-1e5f3226596e)


## Data Preparation
### Handling Missing Values
        df_cleaned = df.dropna()

Kami mengidentifikasi dan menangani nilai-nilai yang hilang (missing values) di dalam dataset. Menggunakan metode dropna(), kami menghapus baris-baris yang memiliki data kosong untuk mencegah pengaruh negatif pada performa model. Langkah ini penting karena data yang hilang dapat menyebabkan bias atau kesalahan dalam prediksi model.
### Handling Outliers
### 1. Menghitung IQR (Interquartile Range)

Q1 = numerical_features.quantile(0.25)
Q3 = numerical_features.quantile(0.75)
IQR = Q3 - Q1

    Q1 (Quartile 1) dan Q3 (Quartile 3) adalah nilai kuartil pertama dan ketiga dari data numerik.
        Q1 adalah nilai di mana 25% data berada di bawahnya.
        Q3 adalah nilai di mana 75% data berada di bawahnya.
    IQR adalah rentang interkuartil yang didapatkan dari pengurangan antara Q3 dan Q1. IQR ini mewakili rentang data utama tanpa memperhitungkan outlier.

### 2. Menentukan Batas Outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

    Lower Bound dan Upper Bound digunakan untuk mendeteksi outlier:
        Lower Bound: Nilai di bawah Q1 - 1.5 * IQR dianggap sebagai outlier bawah.
        Upper Bound: Nilai di atas Q3 + 1.5 * IQR dianggap sebagai outlier atas.
    Faktor 1.5 adalah standar yang digunakan untuk menentukan outlier. Nilai yang berada lebih dari 1.5 kali jarak IQR dianggap sebagai outlier.

### 3. Mendeteksi Baris yang Mengandung Outlier

outliers = ((numerical_features < lower_bound) | (numerical_features > upper_bound)).any(axis=1)

    Logika ini digunakan untuk mendeteksi apakah ada nilai di setiap kolom numerik yang melampaui batas outlier.
    numerical_features < lower_bound mendeteksi data yang lebih kecil dari batas bawah.
    numerical_features > upper_bound mendeteksi data yang lebih besar dari batas atas.
    any(axis=1) mengembalikan True untuk setiap baris yang memiliki outlier di salah satu kolom numerik.

### 4. Menampilkan Baris yang Mengandung Outlier
    df_outliers = df[outliers]
    df[outliers] menyaring dataframe untuk hanya menampilkan baris yang mengandung outlier.
    print(df_outliers) digunakan untuk menampilkan baris-baris tersebut.
   
Metode IQR digunakan untuk mendeteksi outlier dengan menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3). Data yang berada di luar batas lower bound dan upper bound dianggap sebagai outlier. Setelah mendeteksi outlier, baris-baris yang mengandung outlier disaring dan ditampilkan untuk analisis lebih lanjut.
Outlier merupakan data yang menyimpang jauh dari nilai-nilai lain dalam dataset dan bisa mempengaruhi performa model. Kami menggunakan metode statistik seperti Z-score untuk mendeteksi outlier pada fitur numerik. Jika ditemukan nilai-nilai yang ekstrem, kami memutuskan untuk menghilangkannya guna menjaga kualitas data.
### Label Encoding

  
       categorical_encode_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

Kami melakukan proses Label Encoding untuk mengubah variabel kategorikal menjadi numerik. Fitur-fitur seperti Sex, ChestPainType, RestingECG, ExerciseAngina, dan ST_Slope diubah menjadi representasi numerik menggunakan metode ini. Langkah ini penting agar model machine learning dapat memahami data kategorikal dengan lebih baik.
### Feature Engineering

Dilakukan One-Hot Encoding untuk fitur kategori seperti Sex, ChestPainType, RestingECG, ExerciseAngina, dan ST_Slope.


              df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

### Feature Scaling

Fitur numerik seperti Age, RestingBP, Cholesterol, MaxHR, dan Oldpeak dinormalisasi menggunakan StandardScaler untuk menghindari bias pada model.

             from sklearn.preprocessing import StandardScaler
             scaler = StandardScaler()
             df_scaled = scaler.fit_transform(df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])

### Split Data

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
| Model                  | Train Accuracy (%) | Test Accuracy (%) | Precision | Recall | F1 Score |
|------------------------|--------------------|-------------------|-----------|--------|----------|
| K-Nearest Neighbor (KNN) | 95                 | 78                | 0.76      | 0.80   | 0.78     |
| Random Forest (RF)      | 100                | 84                | 0.85      | 0.83   | 0.84     |
| Boosting                | 95                 | 82                | 0.82      | 0.81   | 0.81     |
| Boosting_Tuned          | 96                 | 84                | -         | -      | -        |
| RF_Turned               | 100                | 83                | -         | -      | -        |
Train Accuracy dan Test Accuracy pada tabel di atas didasarkan pada grafik yang menunjukkan nilai akurasi untuk model yang berbeda.
Beberapa nilai Precision, Recall, dan F1 Score tidak tersedia untuk model Boosting_Tuned dan RF_Turned karena nilai tersebut tidak diwakili dalam grafik yang Anda berikan.

## Visualisasi Plot
### Visualisasi MSE untuk Model yang Berbeda

Pada grafik di bawah ini, kita dapat melihat perbandingan **MSE** dari beberapa model yang digunakan. Grafik ini memvisualisasikan MSE baik pada data latih maupun data uji untuk setiap model.
![Plot MSE_model](https://github.com/user-attachments/assets/be30638f-f3fb-4627-a7b3-4c9111900dad)

- **RF_turned** dan **Boosting** memiliki performa yang baik pada data latih dan uji, dengan nilai MSE uji yang rendah.
- **KNN** menunjukkan perbedaan yang signifikan antara MSE latih dan uji, yang mungkin menunjukkan **overfitting**.
- **Random Forest (RF)** pada versi tuning maupun tidak menunjukkan bahwa performanya tetap baik dengan MSE uji yang rendah.

Dari evaluasi ini, dapat disimpulkan bahwa model **Random Forest (RF)** dan **Boosting** memberikan performa yang lebih baik dibandingkan model lainnya dalam hal MSE dan akurasi

### Akurasi Akurasi Model

Grafik di bawah menunjukkan perbandingan **akurasi** dari beberapa model:


![Plot model Akurasi](https://github.com/user-attachments/assets/5ca8e12d-a4eb-48b2-a9de-25e27946aff6)

Dari grafik akurasi tersebut, kita dapat melihat bahwa:

- **Random Forest (RF)**, terutama yang sudah dituning (RF_turned), memiliki **akurasi tertinggi**, baik pada data latih maupun data uji.
- **Boosting** juga memiliki akurasi yang sangat baik, meskipun sedikit di bawah **Random Forest**.
- **K-Nearest Neighbor (KNN)** memiliki akurasi yang lebih rendah dibandingkan model lainnya, baik pada data latih maupun data uji.


## Kesimpulan

   -  Random Forest memberikan hasil terbaik dengan akurasi 84% dan stabil di berbagai metrik evaluasi.
   -  Boosting memberikan hasil yang baik tetapi sedikit di bawah Random Forest.
   -  KNN memiliki akurasi yang lebih rendah, kemungkinan disebabkan oleh overfitting pada data latih.

Dengan demikian, model Random Forest menjadi pilihan terbaik untuk prediksi penyakit jantung dalam proyek ini.

## Referensi:
- Rahman, S. et al. (2022). "Predicting Heart Disease Using Machine Learning Techniques". Journal of Healthcare Engineering.
- World Health Organization (2021). "Cardiovascular diseases (CVDs)". Available at: https://www.who.int.
- aleema S., S. Syed dan M. Ahmad, "Application of Machine Learning Algorithms for Heart Disease Prediction," International Journal of Advanced Computer Science and Applications, vol. 11, no. 4, 2020.
