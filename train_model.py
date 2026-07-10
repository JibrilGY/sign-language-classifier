import pandas as pd
from xgboost import XGBClassifier # 🌟 XGBoost kullanıyoruz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import accuracy_score

# 1. Veriyi Yükle
df = pd.read_csv("tid_dataset_2hands.csv")
X = df.drop('label', axis=1)
y = df['label']

# 2. Etiketleri Sayıya Çevir (XGBoost bunu ister)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Boş Verileri Doldur
imputer = SimpleImputer(strategy='constant', fill_value=0.0)
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)

# 4. XGBoost Modeli (Derinliği arttırıyoruz)
# XGBClassifier'ı şu şekilde ayarla (daha "düşünen" bir model için)
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,       # 🌟 HER EĞİTİMDE VERİNİN %80'İNİ KULLAN
    colsample_bytree=0.8,# 🌟 HER AĞAÇTA ÖZELLİKLERİN %80'İNİ KULLAN
    random_state=42
)
model.fit(X_train, y_train)

# 5. Başarıyı Ölç
y_pred = model.predict(X_test)
print(f"DOĞRULUK ORANI: %{accuracy_score(y_test, y_pred) * 100:.2f}")

# 6. Kaydet
joblib.dump(model, "gesture_mlp_model.pkl") # İsmi aynı tut ki main.py bozulmasın
joblib.dump(imputer, "imputer.pkl")
joblib.dump(le, "label_encoder.pkl") # 🌟 Yeni: Harfleri geri çevirmek için

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig('confusion_matrix.png')
print("Grafik 'confusion_matrix.png' olarak kaydedildi. Dosyayı klasöründen açıp incele.")