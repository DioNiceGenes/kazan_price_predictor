import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import mlflow
import lightgbm as lgb
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import settings
from preprocessing import create_features


class QuantileModels:
    def __init__(self):
        self.models_dir = Path("models/quantile")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dict = None
        self.X_raw = None
        self.y = None
        
        self.results = {}

    def clean_feature_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """Улучшенная очистка названий колонок для LightGBM"""
        X = X.copy()
        
        new_names = {}
        seen = {}
        
        for col in X.columns:
            # Базовая очистка
            clean = str(col)
            clean = clean.replace(" ", "_")
            clean = re.sub(r'[^a-zA-Z0-9_]', '_', clean)
            clean = re.sub(r'_+', '_', clean).strip('_')
            
            # Если после очистки имя стало неуникальным — добавляем суффикс
            if clean in seen:
                counter = seen[clean] + 1
                seen[clean] = counter
                clean = f"{clean}_{counter}"
            else:
                seen[clean] = 0
                
            new_names[col] = clean
        
        X.rename(columns=new_names, inplace=True)
        
        return X

    def load_data(self):
        """Загружает все необходимые данные"""
        print("Загрузка данных...")
        tdf = pd.read_csv(settings.TDF_PATH)
        self.data_dict = create_features(tdf)
        
        preprocessor = self.data_dict["preprocessor"]
        self.X_raw = pd.DataFrame(preprocessor.transform(self.data_dict['X']), columns=self.data_dict['feature_names'], index=self.data_dict["X"].index)
        self.y = self.data_dict['y']
        
        # Загружаем сохранённые признаки
        handcrafted_path = Path("data/processed/handcrafted_features.csv")
        image_path = Path("data/processed/image_embeddings.csv")
        
        self.handcrafted = pd.read_csv(handcrafted_path, index_col=0) if handcrafted_path.exists() else None
        self.image_pca = pd.read_csv(image_path, index_col=0) if image_path.exists() else None
        
        print(f"X_raw: {self.X_raw.shape}")
        if self.handcrafted is not None:
            print(f"Handcrafted: {self.handcrafted.shape}")
        if self.image_pca is not None:
            print(f"Image PCA: {self.image_pca.shape}")

    def prepare_data(self, use_handcrafted=False, use_images=False):
        """Подготавливает данные для конкретной комбинации"""
        X = self.X_raw.copy()
        
        if use_handcrafted and self.handcrafted is not None:
            X = pd.concat([X.reset_index(drop=True), 
                          self.handcrafted.reset_index(drop=True)], axis=1)
        
        if use_images and self.image_pca is not None:
            img_df = pd.DataFrame(
                self.image_pca,
                columns=[f"img_pca_{i}" for i in range(self.image_pca.shape[1])],
                index=self.X_raw.index
            )
            X = pd.concat([X.reset_index(drop=True), img_df.reset_index(drop=True)], axis=1)

        X = self.clean_feature_names(X)
        return X

    def train_quantile_model(self, X: pd.DataFrame, y: pd.Series, quantile: float, name: str):
        """Обучение одной квантильной модели LightGBM"""
        print(f"Обучение Quantile α={quantile:.1f} | {name} | Features: {X.shape[1]}")
        
        params = {
            'objective': 'quantile',
            'alpha': quantile,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 64,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
            'force_row_wise': True
        }
        
        mlflow_model_name = f"lgb_q{int(quantile*10)}_{name}"
        
        with mlflow.start_run(run_name=f"Quantile_{quantile}_{name}"):
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            
            # Логирование
            mlflow.log_param("quantile", quantile)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("dataset", name)
            
            # Сохранение модели на диск
            model_path = self.models_dir / f"{mlflow_model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"Модель сохранена: {model_path.name}")
            
            # Логирование в MLflow
            mlflow.sklearn.log_model(model, artifact_path=mlflow_model_name)
            
        return model

    def evaluate_quantiles(self, X_test, y_test, models_dict, name: str):
        """Оценка качества квантильной регрессии"""
        lower = models_dict[0.1].predict(X_test)
        median = models_dict[0.5].predict(X_test)
        upper = models_dict[0.9].predict(X_test)
        
        mae = mean_absolute_error(y_test, median)
        rmse = np.sqrt(mean_squared_error(y_test, median))
        r2 = r2_score(y_test, median)
        
        # Coverage (процент реальных значений, попавших в интервал)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        
        # Ширина интервала
        interval_width = np.mean(upper - lower)
        
        print(f"\n{'='*80}")
        print(f"ОЦЕНКА МОДЕЛИ: {name}")
        print(f"{'='*80}")
        print(f"MAE (медиана)     : {mae:,.0f} ₽")
        print(f"RMSE              : {rmse:,.0f} ₽")
        print(f"R²                : {r2:.4f}")
        print(f"Coverage 80% PI   : {coverage:.3%}  (цель ≈ 80%)")
        print(f"Средняя ширина интервала: {interval_width:,.0f} ₽")
        print(f"{'='*80}\n")
        
        self.results[name] = {
            "MAE": mae, "RMSE": rmse, "R2": r2, 
            "Coverage": coverage, "Interval_Width": interval_width
        }

    def train_all(self):
        self.load_data()
        
        combinations = [
            ("tabular_only", False, False),
            ("tabular_handcrafted", True, False),
            ("tabular_images", False, True),
            ("tabular_handcrafted_images", True, True)
        ]
        
        for name, use_hc, use_img in combinations:
            X = self.prepare_data(use_handcrafted=use_hc, use_images=use_img)
            
            models_dict = {}
            for q in [0.1, 0.5, 0.9]:
                models_dict[q] = self.train_quantile_model(X, self.y, q, name)
            
            # Оценка на всех данных (можно позже сделать на hold-out)
            self.evaluate_quantiles(X, self.y, models_dict, name)

        # Финальная таблица
        comparison = pd.DataFrame(self.results).T
        print("\n" + "="*100)
        print("ФИНАЛЬНОЕ СРАВНЕНИЕ КВАНТИЛЬНЫХ МОДЕЛЕЙ")
        print("="*100)
        print(comparison.round(4))
        
        comparison.to_csv("reports/quantile_models_comparison.csv")
        print(f"\n Все модели сохранены в {self.models_dir}")


if __name__ == "__main__":
    qmodels = QuantileModels()
    qmodels.train_all()