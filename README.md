
# 🧠 Medical Diagnosis Using AI

An AI-powered web application that helps predict the presence of critical diseases — including **Diabetes**, **Heart Disease**, **Parkinson's Disease**, **Lung Cancer**, and **Thyroid Disorders** — using machine learning models trained on real medical datasets.

---

## 🔍 Problem Statement

In many low-resource or rural regions, access to timely and accurate diagnosis is limited due to a lack of medical infrastructure, skilled professionals, and expensive lab-based tests. This leads to **late-stage detection** of serious diseases, resulting in higher morbidity and mortality.

---

## ✅ Proposed Solution

This project provides a **multi-disease prediction platform** using trained ML models deployed via a user-friendly **Streamlit web application**. It allows users to input basic health indicators and receive instant predictive feedback for early awareness and medical guidance.

---

## 💡 Features

- 🔢 Input-based prediction for five diseases
- ⚙️ Pretrained ML models (scikit-learn)
- 🌐 Browser-based, no installation needed
- 📱 Mobile-responsive UI
- 🧬 Built with Streamlit for rapid deployment

---

## 🛠️ Technology Stack

| Layer       | Tools Used                                   |
|-------------|----------------------------------------------|
| **Frontend**| Streamlit, CSS (injected)                    |
| **Backend** | Python, scikit-learn, Pandas, NumPy          |
| **Models**  | Logistic Regression, Random Forest, SVM, KNN |
| **Deployment** | Streamlit Cloud + GitHub                   |

---

## 📂 Project Structure

```
📁 medical-diagnosis-ai/
├── app.py                    # Main Streamlit application
├── Models/                  # Pretrained .sav ML models
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   ├── parkinsons_model.sav
│   ├── lungs_disease_model.sav
│   └── Thyroid_model.sav
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
```

---

## 🚀 How to Run Locally

1. **Clone this repository**
```bash
git clone https://github.com/arshadibrahim882/Medical-Diagnosis-using-AI
cd medical-diagnosis-ai
```

2. **Create a virtual environment (optional)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app can be deployed for free using **[Streamlit Cloud](https://streamlit.io/cloud)**:

1. Push this repo to GitHub
2. Go to Streamlit Cloud and sign in
3. Click **“New App”** → select your repo
4. Set the main file as `app.py`
5. Click **Deploy**

You will get a shareable public URL like:
```
https://yourname-medical-diagnosis.streamlit.app
```

---

## 📈 Future Scope

- Add PDF reports and patient history tracking
- Support image-based models (e.g., X-rays)
- Integrate wearable sensor data (IoT)
- Role-based login for doctors and patients
- Cloud database for storing predictions

---

## 📚 References

- [UCI Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- [Lung Cancer Dataset (Kaggle)](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer-dataset)
- [Thyroid Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/thyroid+disease)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 👤 Author

Created by [**Sheik Arshad Ibrahim**](https://github.com/arshadibrahim882)  
© 2025

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
