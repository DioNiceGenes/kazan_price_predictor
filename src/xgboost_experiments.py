import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from pathlib import Path
import time
import mlflow
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import settings
from preprocessing import create_features


class XGBoostExperiments:
    def __init__(self):
        self.models_dir = Path("models/xgboost")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = Path("reports/xgboost_analysis")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dict = None
        self.X_raw = None
        self.y = None
        
        self.results = {}

    def load_data(self):
        """Загружает все необходимые данные"""
        print("Загрузка данных...")
        tdf = pd.read_csv(settings.TDF_PATH)
        self.data_dict = create_features(tdf)
        
        preprocessor = self.data_dict["preprocessor"]
        self.X_raw = pd.DataFrame(preprocessor.transform(self.data_dict['X']), columns=preprocessor.get_feature_names_out(), index=self.data_dict["X"].index)
        self.y = self.data_dict['y']
        
        # Загружаем сохранённые признаки
        handcrafted_path = Path("data/processed/handcrafted_features.csv")
        image_path = Path("data/processed/image_embeddings.csv")
        
        self.handcrafted = pd.read_csv(handcrafted_path, index_col=0) if handcrafted_path.exists() else None
        self.image_pca = pd.read_csv(image_path, index_col=0) if image_path.exists() else None
        print("Extra features exist: ", self.handcrafted is not None, self.image_pca is not None)
        
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

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def plot_learning_curve(self, model, X, y, name: str):
        """Сохраняет кривую обучения"""
        print(f"Построение кривой обучения для {name}...")
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            cv=2,
            scoring='neg_mean_absolute_error',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        
        train_mean = -np.mean(train_scores, axis=1)
        train_std = -np.std(train_scores, axis=1)
        val_mean = -np.mean(val_scores, axis=1)
        val_std = -np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training MAE")
        plt.plot(train_sizes, val_mean, 'o-', color="g", label="Validation MAE")
        plt.xlabel("Training examples")
        plt.ylabel("MAE (lower is better)")
        plt.title(f"Learning Curve — {name}")
        plt.legend(loc="best")
        plt.grid(True)
        
        plt.savefig(self.figures_dir / f"learning_curve_{name}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, model, X, name: str, top_n=30):
        """Сохраняет график важности признаков"""
        print(f"Построение Feature Importance для {name}...")
        importances = model.feature_importances_
        features = X.columns
        
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feat_imp.values[:top_n], y=feat_imp.index[:top_n])
        plt.title(f"Top {top_n} Feature Importance — {name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        
        plt.savefig(self.figures_dir / f"feature_importance_{name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        time.sleep(10)
        plt.close()

    def run_grid_search(self, X_train, y_train, X_test, y_test, name: str):
        """GridSearch + обучение + визуализации"""
        print(f"\n{'='*90}")
        print(f"XGBoost GridSearch → {name}")
        print(f"{'='*90}")
        
        param_grid = {
            'n_estimators': [800],
            'max_depth': [10],
            'learning_rate': [0.05],
            'subsample': [0.8],
        }
        
        base_model = xgb.XGBRegressor(
            random_state=42,
            tree_method='hist',
            objective='reg:squarederror'
        )
        
        with mlflow.start_run(run_name=f"XGBoost_GridSearch_{name}"):
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Метрики
            y_pred = best_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.xgboost.log_model(best_model, "best_model")
            
            # Сохранение модели
            model_path = self.models_dir / f"xgb_{name}.pkl"
            joblib.dump(best_model, model_path)
            
            print(f"Лучшая модель сохранена: {model_path.name}")
            print(f"MAE : {mae:,.0f} ₽")
            print(f"RMSE: {rmse:,.0f} ₽")
            print(f"R²  : {r2:.4f}")
            
            # Визуализации
            self.plot_learning_curve(best_model, X_train, y_train, name)
            self.plot_feature_importance(best_model, X_train, name, top_n=25)
            
            self.results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
            
        return best_model

    def run_all_experiments(self):
        self.load_data()
        
        combinations = [
            ("tabular_only", False, False),
            ("tabular_handcrafted", True, False),
            ("tabular_images", False, True),
            ("tabular_handcrafted_images", True, True)
        ]
        
        for name, use_hc, use_img in combinations:
            X_train, X_test, y_train, y_test = self.prepare_data(use_handcrafted=use_hc, use_images=use_img)
            print(X_train.shape)
            self.run_grid_search(X_train, y_train, X_test, y_test, name)
        
        # Финальная таблица
        comparison = pd.DataFrame(self.results).T
        print("\n" + "="*100)
        print("ФИНАЛЬНОЕ СРАВНЕНИЕ XGBoost МОДЕЛЕЙ")
        print("="*100)
        print(comparison[['MAE', 'RMSE', 'R2']].round(4))
        
        comparison.to_csv("reports/xgboost_experiments_comparison.csv")
        print(f"\nВсе модели сохранены в {self.models_dir}")


if __name__ == "__main__":
    exp = XGBoostExperiments()
    exp.run_all_experiments()
