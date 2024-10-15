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

### Fitur-Fitur:
- **Age**: Usia pasien.
- **Sex**: Jenis kelamin pasien.
- **ChestPainType**: Tipe nyeri dada yang dialami pasien.
- **RestingBP**: Tekanan darah saat istirahat.
- **Cholesterol**: Tingkat kolesterol dalam darah.
- **FastingBS**: Gula darah setelah berpuasa.
- **RestingECG**: Hasil rekam jantung saat istirahat.
- **MaxHR**: Denyut jantung maksimum.
- **ExerciseAngina**: Angina yang dipicu oleh olahraga.
- **Oldpeak**: Penurunan ST setelah olahraga.
- **ST_Slope**: Kemiringan segmen ST saat uji stres jantung.
- **HeartDisease**: Label target, menunjukkan apakah pasien menderita penyakit jantung (1) atau tidak (0).

Analisis eksplorasi data dilakukan untuk memahami distribusi dari setiap fitur, termasuk penggunaan visualisasi seperti histogram dan pair plot untuk melihat hubungan antar fitur.
Data Preparation

Beberapa teknik data preparation yang diterapkan termasuk:

Mengatasi missing values.
Normalisasi data pada fitur numerik untuk memudahkan proses pelatihan model.
Split data menjadi 80% data latih dan 20% data uji untuk memastikan bahwa model dapat diuji dengan benar.

Tahapan ini diperlukan agar model yang dibangun dapat dilatih dengan data yang bersih dan memastikan bahwa hasil evaluasi lebih akurat.

Data Preparation

Setelah memahami data, kami melakukan beberapa tahapan Data Preparation untuk mempersiapkan data sebelum model diimplementasikan. Langkah-langkah ini termasuk:

    Handling Missing Values Berdasarkan hasil eksplorasi, tidak ada missing values dalam dataset ini sehingga tidak diperlukan proses imputasi atau penghapusan data.

    Encoding Fitur Kategori Untuk fitur kategori seperti Sex, ChestPainType, RestingECG, ExerciseAngina, dan ST_Slope, kami melakukan encoding agar data bisa digunakan oleh model machine learning. Kami menggunakan One-Hot Encoding untuk fitur dengan lebih dari dua kategori.

    python

df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

Feature Scaling Kami menggunakan StandardScaler untuk menormalisasi fitur numerik seperti Age, RestingBP, Cholesterol, MaxHR, dan Oldpeak agar memiliki skala yang sama dan memudahkan algoritma untuk bekerja lebih baik.

python

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])

Train-Test Split Kami membagi dataset menjadi data latih (80%) dan data uji (20%) untuk memastikan evaluasi model dilakukan pada data yang belum pernah dilihat model.

python

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

## Referensi:

Rahman, S. et al. (2022). "Predicting Heart Disease Using Machine Learning Techniques". Journal of Healthcare Engineering.
World Health Organization (2021). "Cardiovascular diseases (CVDs)".
aleema S., S. Syed dan M. Ahmad, "Application of Machine Learning Algorithms for Heart Disease Prediction," International Journal of Advanced Computer Science and Applications, vol. 11, no. 4, 2020.
