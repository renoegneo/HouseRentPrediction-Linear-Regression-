import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATAFILE = "House_Rent_Dataset.csv"
ARTIFACTS_FILE = "model_artifacts.joblib"

def clean_floor(value):
    try:
        parts = str(value).lower().split(" out of ")
        curr = 0 if "ground" in parts[0] else int(parts[0])
        total = int(parts[1])
        return pd.Series([curr, total])
    except:
        return pd.Series([None, None])

def visualize_regression(y_test, y_pred, r2):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    
    lims = [0, max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='Идеальное предсказание')
    
    plt.xlabel('Фактические значения (Rent)', fontsize=12)
    plt.ylabel('Предсказанные значения', fontsize=12)
    plt.title(f'Регрессионный анализ (R² = {r2:.3f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_plot.png', dpi=500, bbox_inches='tight')
    print("График регрессии сохранён в 'regression_plot.png'")
    plt.show()

def main():
    df = pd.read_csv(DATAFILE)

    if "Floor" in df.columns:
        df[["CurrentFloor", "TotalFloors"]] = df["Floor"].apply(clean_floor)
    
    cols_to_drop = ["Floor", "Area Locality", "Posted On"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).dropna()

    if "Rent" in df.columns:
        df = df[df["Rent"] < df["Rent"].quantile(0.92)] 

    df = pd.get_dummies(df, drop_first=True)
    X = df.drop(columns=["Rent"])
    y = df["Rent"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Обучение завершено.\nTest R^2: {round(r2, 4)}")
    
    visualize_regression(y_test, y_pred, r2)

    artifacts = {
        "model": model,
        "feature_names": X.columns.tolist(),
        "X_test": X_test,
        "y_test": y_test,
    }
    joblib.dump(artifacts, ARTIFACTS_FILE)
    print(f"Артефакты сохранены в '{ARTIFACTS_FILE}'")

if __name__ == "__main__":
    main()