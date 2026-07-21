#Veri Yükleme, Bölme ve Ölçekleme
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

CSV_FILENAME = "tid_dataset_2hands.csv"
df= pd.read_csv(CSV_FILENAME)

X = df.drop(columns=['label']).values
y_raw = df['label'].values

#XGBoost İçin
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#GridSearchCV
from sklearn.model_selection import GridSearchCV
#Aday Algoritmalar
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

models_and_grids = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=5000, random_state=30),
        "params": {"C": [0.1,1,10], "solver": ["lbfgs","saga"]},
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=30),
        "params": {"max_depth": [None,50,100], "min_samples_split":[2,5]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=30),
        "params": {"n_estimators": [50,100], "max_depth": [None,10]}
    },
    "SVM": {
        "model": SVC(random_state=30),
        "params": {"C": [0.1,1,10], "kernel": ["linear","rbf"]}
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3,5,7], "weights": ["uniform","distance"]}
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=30, eval_metric="logloss",),
        "params": {"n_estimators": [50,100], "learning_rate": [0.01,0.1]}
    }
}

best_overall_score = 0
best_overall_model = None
best_overall_name = ""
best_overall_params = {}

for name, info in models_and_grids.items():
    print(f"\n[Yarışma] {name} test ediliyor.")

    grid_search = GridSearchCV(
        estimator=info['model'],
        param_grid=info['params'],
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train_scaled, y_train)

    print(f"-> {name} En İyi Skoru: {grid_search.best_score_:.4f}")
    print(f"-> En İyi Parametreler: {grid_search.best_params_}")

    #Daha iyisine göre güncelleme
    if grid_search.best_score_ > best_overall_score:
        best_overall_score = grid_search.best_score_
        best_overall_model = grid_search.best_estimator_
        best_overall_name = name
        best_overall_params = grid_search.best_params_

print("\n" + "=" * 40)
print(f"🏆 LİGİN ŞAMPİYONU: {best_overall_name}")
print(f"Doğruluk Skoru: {best_overall_score:.4f}")
print(f"Kazanan Parametreler: {best_overall_params}")
print("=" * 40)


import joblib
# Şampiyon modeli, scaler'ı ve label encoder'ı diske kaydet
joblib.dump(best_overall_model, "best_hand_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel, Scaler ve Label Encoder başarıyla kaydedildi!")

# Test seti üzerinde son performans kontrolü
from sklearn.metrics import accuracy_score, classification_report

y_pred = best_overall_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Test Seti Doğruluk Skoru: {test_accuracy:.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))