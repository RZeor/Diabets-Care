from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import mysql.connector 
import os
import joblib  # Use joblib instead of pickle for saving models
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import io
import base64

app = Flask(__name__)
app.secret_key = "secret_key"  # Diperlukan untuk menampilkan pesan flash
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder 'uploads' dan 'models' jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Path untuk menyimpan model
MODEL_PATH = os.path.join(MODEL_FOLDER, "random_forest_model.pkl")

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk halaman form
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/evaluation', methods=['GET', 'POST'])
def evaluation():
    if request.method == 'POST':
        try:
            # Koneksi ke MySQL
            conn = mysql.connector.connect(
                host="localhost",  # Ganti dengan host Anda
                user="root",       # Ganti dengan user Anda
                password="",  # Ganti dengan password Anda
                database="diabetes"  # Nama database
            )

            # Query untuk mengambil data
            query = "SELECT * FROM diabetes_data_upload"
            df = pd.read_sql(query, conn)

            # Pastikan koneksi ditutup setelah selesai
            conn.close()

            # Ganti nilai kategori menjadi numerik
            df_encoded = df.copy()
            for column in df_encoded.columns:
                if df_encoded[column].dtype == 'object':
                    df_encoded[column] = df_encoded[column].astype('category').cat.codes

            # Kolom yang diperlukan untuk fitur dan target
            required_columns = [
                "Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
                "weakness", "Polyphagia", "Genital thrush", "visual blurring",
                "Itching", "Irritability", "delayed healing", "partial paresis",
                "muscle stiffness", "Alopecia", "Obesity", "class"
            ]
            if not all(column in df_encoded.columns for column in required_columns):
                flash("Database does not contain all required columns.")
                return render_template('index.html', df=None, accuracy=None, precision=None, f1=None, cm_image=None)

            # Pisahkan fitur (X) dan target (y)
            X = df_encoded[required_columns[:-1]]  # Semua kolom kecuali 'class'
            y = df_encoded['class']  # Kolom 'class'

            # Bagi data menjadi training dan testing (test_size = 0.3)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # periksa model Random Forest
            model = joblib.load(MODEL_PATH)
            flash("Model loaded from existing file.")

            # Evaluasi model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)

            # Buat grafik Confusion Matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")

            # Konversi gambar ke Base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            cm_image = base64.b64encode(img.getvalue()).decode('utf-8')
            plt.close()

            # Ambil hanya 25 baris pertama untuk ditampilkan
            df_limited = df.head(50)

            flash(f"Model trained successfully! Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}")

            # Tampilkan hasil di halaman training
            return render_template(
                'evaluation.html',
                df=df_limited.to_html(classes='data', header="true", index=False),
                accuracy=accuracy, precision=precision, f1=f1,
                cm_image=cm_image
            )

        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            return render_template('evaluation.html', df=None, accuracy=None, precision=None, f1=None, cm_image=None)

    return render_template('evaluation.html', df=None, accuracy=None, precision=None, f1=None, cm_image=None)

# route untuk halaman predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        data = request.form
        
        # Helper function to map 'Yes' and 'No' to 1 and 0
        def yes_no_to_int(value):
            return 1 if value == 'Yes' else 0

        features = [
            float(data['age']),
            yes_no_to_int(data['gender']),
            yes_no_to_int(data['polyuria']),
            yes_no_to_int(data['polydipsia']),
            yes_no_to_int(data['weight_loss']),
            yes_no_to_int(data['weakness']),
            yes_no_to_int(data['polyphagia']),
            yes_no_to_int(data['genital_thrush']),
            yes_no_to_int(data['visual_blurring']),
            yes_no_to_int(data['itching']),
            yes_no_to_int(data['irritability']),
            yes_no_to_int(data['delayed_healing']),
            yes_no_to_int(data['partial_paresis']),
            yes_no_to_int(data['muscle_stiffness']),
            yes_no_to_int(data['alopecia']),
            yes_no_to_int(data['obesity']),
        ]

        # Prediksi dengan model Random Forest
        model = joblib.load(MODEL_PATH)
        prediction = model.predict([features])[0]
        result = "Positive Diabetes" if prediction == 1 else "Negative Diabetes"
        resulte = "Anda positive diabetes, segera konsultasikan dengan dokter sebelum terlambat" if prediction == 1 else "Anda negative diabetes, tetap jaga pola hidup sehat anda untuk mencegah terkena diabetes di masa depan"

        # Tampilkan hasil di halaman prediction.html
        return render_template('prediction.html', result=result, resulte=resulte)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
