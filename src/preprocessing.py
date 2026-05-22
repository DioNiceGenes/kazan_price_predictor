import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib

from config import settings


def load_processed_data():
    """Загружает предобработанные данные"""
    tdf = pd.read_csv(settings.TDF_PATH)
    print(f"Загружен tdf: {tdf.shape}")
    return tdf


def create_features(tdf: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Создаёт признаки, разделяет на train/test, обучает preprocessor только на train.
    Возвращает DataFrame с именами колонок.
    """
    print("Создание признаков и разделение на train/test...")

    # Целевая переменная
    y = tdf['price'].copy()
    X = tdf.drop(columns=['price']).copy()

    numeric_features = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Числовых признаков: {len(numeric_features)}")
    print(f"Числовые признаки: {numeric_features}")
    print(f"Категориальных признаков: {len(categorical_features)}")
    print(f"Категориальные признаки: {categorical_features}")

    # разделение на train, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Обучаем preprocessor на train
    print("Обучение Preprocessor на train данных...")
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Применяем к test
    X_test_transformed = preprocessor.transform(X_test)

    # Получаем названия колонок после OneHotEncoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(cat_feature_names)

    X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names, index=X_test.index)
    X_full_df = pd.concat([X_train_df, X_test_df]).sort_index()

    print(f"Финальная размерность признаков: {X_full_df.shape}")

    return {
        'X': X,
        'y': y,
        'X_full_df': X_full_df,
        'X_train': X_train_df,
        'y_train': y_train,
        'X_test': X_test_df,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': all_feature_names
    }


def save_artifacts(data_dict):
    """Сохраняет все артефакты"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(data_dict['preprocessor'], models_dir / "preprocessor.pkl")
    joblib.dump(data_dict['X_train'], models_dir / "X_train.pkl")
    joblib.dump(data_dict['X_test'], models_dir / "X_test.pkl")
    joblib.dump(data_dict['y_train'], models_dir / "y_train.pkl")
    joblib.dump(data_dict['y_test'], models_dir / "y_test.pkl")
    
    # Сохраняем названия признаков
    pd.Series(data_dict['feature_names']).to_csv(models_dir / "feature_names.csv", index=False)
    
    print(f" Артефакты успешно сохранены в {models_dir}/")


if __name__ == "__main__":
    tdf = load_processed_data()
    data_dict = create_features(tdf, test_size=0.2, random_state=42)
    save_artifacts(data_dict)
    print("\n Подготовка признаков завершена")