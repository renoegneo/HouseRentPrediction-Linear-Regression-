import joblib
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache

ARTIFACTS_FILE = "model_artifacts.joblib"

@lru_cache(maxsize=1)
def load_artifacts():
    art = joblib.load(ARTIFACTS_FILE)
    return art["model"], art["feature_names"], art["X_test"], art["y_test"]

def get_r2_score():
    model, _, X_test, y_test = load_artifacts()
    return float(model.score(X_test, y_test))

def predict_from_input(data):
    model, features, _, _ = load_artifacts()
    X_single = build_predict_row(data, features)
    return float(model.predict(X_single)[0])

def clean_floor(value):
    try:
        parts = str(value).lower().split(" out of ")
        curr = 0 if "ground" in parts[0] else int(parts[0])
        total = int(parts[1])
        return curr, total
    except:
        return 0, 0

def build_predict_row(data, feature_names):
    curr, total = clean_floor(data.get("Floor"))
    data["CurrentFloor"], data["TotalFloors"] = curr, total
    
    row = pd.DataFrame(0, index=[0], columns=feature_names)
    
    for col in feature_names:
        if col in data:
            row.loc[0, col] = data[col]
        elif "_" in col:
            cat_name, val = col.split("_", 1)
            if str(data.get(cat_name)) == val:
                row.loc[0, col] = 1
    return row

def main():
    try:
        model, features, X_test, y_test = load_artifacts()
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    y_pred = model.predict(X_test)
    print(f"Test R^2: {model.score(X_test, y_test):.4f}")


    size = int(input("Введите размер квартиры (Size): "))
    bhk = int(input("Введите количество комнат (BHK): "))
    bathroom = int(input("Введите количество ванных комнат (Bathroom): "))
    floor = input("Введите этаж (Floor, напр'1 out of 5'): ")
    city = input("Введите город (City): ")
    area_type = input("Введите тип площади (Area Type, например 'Super Area'): ")
    furnishing_status = input("Введите статус меблировки (Furnishing Status, например 'Semi-Furnished'): ")

    new_house = {
        "Size": size, "BHK": bhk, "Bathroom": bathroom,
        "Floor": floor, "City": city,
        "Area Type": area_type, "Furnishing Status": furnishing_status
    }
    
    prediction = predict_from_input(new_house)
    print(f"\nПредсказанная стоимость: {prediction:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color="powderblue", alpha=0.4, label="Тестовые данные")
    plt.scatter([prediction], [prediction], color="red", s=100, edgecolors='black', label="Новый прогноз")
    
    lims = [0, max(y_test.max(), prediction)]
    plt.plot(lims, lims, color="black", linestyle="--", alpha=0.5)

    plt.xlabel("Реальная цена")
    plt.ylabel("Предсказанная цена")
    plt.title("Результаты Линейной Регрессии")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    main()
