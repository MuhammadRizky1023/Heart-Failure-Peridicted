Laporan Proyek Machine Learning - Muhammad Rizky
Domain Proyek

Penyakit jantung merupakan salah satu penyebab utama kematian di dunia. Dengan adanya perkembangan teknologi dan data medis yang besar, machine learning dapat digunakan untuk membantu memprediksi risiko seseorang terkena penyakit jantung berdasarkan faktor-faktor seperti usia, jenis kelamin, tekanan darah, kadar kolesterol, dan lainnya.

Pentingnya prediksi ini adalah untuk membantu klinik dan rumah sakit dalam memberikan perawatan lebih awal dan mencegah risiko fatal bagi pasien.

Business Understanding
Problem Statements

    Bagaimana cara memprediksi penyakit jantung dengan menggunakan data medis seperti usia, tekanan darah, dan kadar kolesterol?
    Algoritma apa yang memberikan hasil prediksi terbaik untuk kasus penyakit jantung?

Goals

    Membangun model machine learning yang mampu memprediksi apakah seseorang memiliki risiko terkena penyakit jantung atau tidak.
    Membandingkan performa beberapa algoritma seperti K-Nearest Neighbor (KNN), Random Forest, dan Boosting, untuk menemukan model terbaik.

Solution Statements

    Kami akan menggunakan tiga algoritma: K-Nearest Neighbor (KNN), Random Forest, dan Boosting untuk memecahkan masalah prediksi.
    Setelah membangun model dasar, hyperparameter tuning dilakukan untuk meningkatkan performa model, terutama pada algoritma Random Forest dan Boosting.

Data Understanding

Dataset yang digunakan adalah "Heart Failure Prediction" dari Kaggle. Dataset ini terdiri dari 918 sampel dengan 12 fitur yang menjelaskan karakteristik pasien, termasuk usia, jenis kelamin, dan kondisi medis lainnya.

Fitur dalam dataset:

    Age: Usia pasien.
    Sex: Jenis kelamin pasien.
    ChestPainType: Jenis nyeri dada yang dialami.
    Cholesterol: Kadar kolesterol pasien.
    HeartDisease: Target label, apakah pasien mengalami penyakit jantung (1) atau tidak (0).

Analisis eksplorasi data dilakukan untuk memahami distribusi dari setiap fitur, termasuk penggunaan visualisasi seperti histogram dan pair plot untuk melihat hubungan antar fitur.
Data Preparation

Beberapa teknik data preparation yang diterapkan termasuk:

    Mengatasi missing values.
    Normalisasi data pada fitur numerik untuk memudahkan proses pelatihan model.
    Split data menjadi 80% data latih dan 20% data uji untuk memastikan bahwa model dapat diuji dengan benar.

Tahapan ini diperlukan agar model yang dibangun dapat dilatih dengan data yang bersih dan memastikan bahwa hasil evaluasi lebih akurat.
Modeling

Pada bagian ini, tiga model telah dibangun dan diuji:

    K-Nearest Neighbor (KNN): Model sederhana yang bekerja dengan mengukur jarak antara sampel dan menentukan kelas berdasarkan mayoritas tetangga terdekat.
    Random Forest (RF): Model ensemble yang membangun banyak pohon keputusan dan mengambil hasil berdasarkan voting dari tiap pohon.
    Boosting: Model yang berfokus pada memperbaiki prediksi yang salah dari model sebelumnya.

Improvement: Hyperparameter tuning dilakukan pada Random Forest dan Boosting untuk memaksimalkan akurasi model dengan menggunakan grid search. Setelah tuning, Random Forest menunjukkan hasil yang paling baik dengan performa yang stabil.
Evaluation

Model dievaluasi menggunakan metrik Mean Squared Error (MSE) dan akurasi. Berikut hasil dari setiap model:

    K-Nearest Neighbor (KNN):
        MSE Latih: 0.00012, MSE Uji: 0.00037
        Akurasi: 78%
    Random Forest (RF):
        MSE Latih: 0.0, MSE Uji: 0.000158
        Akurasi: 84%
    Boosting:
        MSE Latih: 0.000061, MSE Uji: 0.000141
        Akurasi: 82%

Random Forest dipilih sebagai model terbaik karena memiliki nilai MSE yang rendah dan akurasi yang lebih baik dibandingkan model lain.

Referensi:

    Rahman, S. et al. (2022). "Predicting Heart Disease Using Machine Learning Techniques". Journal of Healthcare Engineering.
    World Health Organization (2021). "Cardiovascular diseases (CVDs)".
