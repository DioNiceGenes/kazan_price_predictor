import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
import mlflow
import mlflow.sklearn
import xgboost as xgb

from config import settings
from preprocessing import load_processed_data, create_features


plt.style.use('seaborn-v0_8-darkgrid')


class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.figures_dir = Path("reports/figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self, model, model_name: str):
        """Оценивает модель на test и логирует в MLflow"""
        y_pred = model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        print(f"{model_name:20} | MAE: {mae:8,.0f} ₽ | RMSE: {rmse:8,.0f} ₽ | R²: {r2:.4f}")

        return {"mae": mae, "rmse": rmse, "r2": r2}
    
    def evaluate_xgboost(self, 
                        X=None, y=None,
                        test_size=0.2,
                        random_state=42,
                        run_name=None,
                        tags=None,
                        params=None,
                        log_feature_importance=True,
                        log_model=True):
        """
        Улучшенная версия evaluate_xgboost внутри ModelTrainer.
        """
        X = X if X is not None else self.X_train
        y = y if y is not None else self.y_train
        
        run_name = run_name or "XGBoost_Model"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Разделение данных
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Гиперпараметры
            default_params = {
                "n_estimators": 600,
                "max_depth": 9,
                "learning_rate": 0.1,
                "random_state": random_state,
                "tree_method": "hist",
                "enable_categorical": True
            }
            
            model_params = params if params is not None else default_params
            
            mlflow.log_params(model_params)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            
            if tags:
                mlflow.set_tags(tags)
            mlflow.set_tag("model", "XGBoost")
            mlflow.set_tag("data_shape", f"{X.shape[0]}x{X.shape[1]}")
            
            # Обучение
            model = xgb.XGBRegressor(**model_params)
            model.fit(X_train, y_train)
            
            # Предсказание и метрики
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            # Логирование
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mae_thousands", mae / 1000)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            if log_model:
                mlflow.xgboost.log_model(model, "xgboost_model")
            
            # Feature Importance
            if log_feature_importance and hasattr(X, 'columns'):
                plt.figure(figsize=(12, 8))
                importance = model.feature_importances_
                feat_imp = pd.Series(importance, index=X.columns).nlargest(30)
                
                sns.barplot(x=feat_imp.values, y=feat_imp.index)
                plt.title(f"Top 30 Feature Importance — {run_name}")
                plt.tight_layout()
                
                fi_path = f"feature_importance_{run_name.lower().replace(' ', '_')}.png"
                plt.savefig(self.figures_dir / fi_path, dpi=200, bbox_inches='tight')
                plt.show()
                mlflow.log_artifact(str(self.figures_dir / fi_path))
                plt.close()
            
            # Вывод
            print(f"\n{'='*75}")
            print(f" Run: {run_name}")
            print(f" MAE : {mae:,.0f} ₽  ({mae/1000:.1f} тыс.₽)")
            print(f" RMSE: {rmse:,.0f} ₽")
            print(f" R²  : {r2:.4f}")
            print(f" Run ID: {run.info.run_id}")
            print(f"{'='*75}\n")
            
            return mae, r2, model

    def train_baseline_models(self):
        """Обучает и сравнивает базовые модели"""
        print("\nОбучение базовых моделей")
        
        models = {
            "LinearRegression": LinearRegression(),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1)
        }

        results = {}

        with mlflow.start_run(run_name="Baseline_Models_Comparison"):
            for name, model in models.items():
                print(f"Обучение {name}...")
                model.fit(self.X_train, self.y_train)
                metrics = self.evaluate_model(model, name)
                results[name] = metrics
                
                # Логируем в MLflow
                mlflow.log_metric(f"{name}_mae", metrics["mae"])
                mlflow.log_metric(f"{name}_rmse", metrics["rmse"])
                mlflow.log_metric(f"{name}_r2", metrics["r2"])

        # Сравнительная таблица
        comparison = pd.DataFrame(results).T
        comparison = comparison.sort_values("mae")
        print("Сравнение базовых моделей")
        print(comparison.round(4))
        
        return comparison

    def run_grid_search(self, estimator, param_grid, model_name="XGBoost", run_name=None):
        """GridSearchCV с логированием"""
        run_name = run_name or f"GridSearch_{model_name}"
        
        with mlflow.start_run(run_name=run_name) as run:
            print(f"\n GridSearchCV для {model_name}...")
            
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_rmse = np.sqrt(-grid_search.best_score_)
            
            # Метрики на test
            y_pred = best_model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            
            # Логирование
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_rmse", best_cv_rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_r2", r2)
            mlflow.sklearn.log_model(best_model, "best_model")
            
            print(f" Best params: {best_params}")
            print(f" Test MAE: {mae:,.0f} ₽ | RMSE: {rmse:,.0f} ₽ | R²: {r2:.4f}")
            
            return best_model, best_params, grid_search

    def plot_and_log_learning_curve(self, model, model_name="Best_Model"):
        """Learning Curve"""
        print(f" Построение Learning Curve для {model_name}...")
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, self.X_train, self.y_train,
            cv=5,
            scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 12),
            n_jobs=-1
        )
        
        train_mean = -np.mean(train_scores, axis=1)
        train_std = -np.std(train_scores, axis=1)
        val_mean = -np.mean(val_scores, axis=1)
        val_std = -np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, val_mean, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("MSE (lower is better)")
        plt.title(f"Learning Curve — {model_name}")
        plt.legend(loc="best")
        plt.grid(True)
        
        filename = f"learning_curve_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(self.figures_dir / filename, dpi=200, bbox_inches='tight')
        plt.show()
        
        mlflow.log_artifact(str(self.figures_dir / filename))
        plt.close()
        print(f"Learning Curve сохранён: {filename}")

    def plot_and_log_feature_importance(self, model, model_name="Model", top_n=30):
        """Feature Importance"""
        print(f" Feature Importance для {model_name}...")
        
        importances = model.feature_importances_
        features = self.X_train.columns if hasattr(self.X_train, 'columns') else [f"feat_{i}" for i in range(len(importances))]
        
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        plt.barh(feat_imp.index[:top_n], feat_imp.values[:top_n])
        plt.title(f"Top {top_n} Feature Importance — {model_name}")
        plt.xlabel("Importance")
        plt.gca().invert_yaxis()
        
        filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(self.figures_dir / filename, dpi=200, bbox_inches='tight')
        plt.show()
        
        mlflow.log_artifact(str(self.figures_dir / filename))
        plt.close()
        print(f"Feature Importance сохранён: {filename}")


if __name__ == "__main__":    
    tdf = load_processed_data()
    data_dict = create_features(tdf)
    
    trainer = ModelTrainer(
        data_dict['X_train'], 
        data_dict['y_train'], 
        data_dict['X_test'], 
        data_dict['y_test']
    )
    
    # 1. Базовые модели
    comparison = trainer.train_baseline_models()
    
    # 2. GridSearch для XGBoost
    xgb_param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9]
    }
    
    best_xgb, best_params, _ = trainer.run_grid_search(
        estimator=xgb.XGBRegressor(random_state=42, tree_method='hist'),
        param_grid=xgb_param_grid,
        model_name="XGBoost",
        run_name="XGBoost_GridSearch_Final"
    )
    
    # 3. Визуализации
    trainer.plot_and_log_learning_curve(best_xgb, "XGBoost")
    trainer.plot_and_log_feature_importance(best_xgb, "XGBoost", top_n=30)
