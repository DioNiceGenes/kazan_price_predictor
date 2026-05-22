import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from torchvision import models
from backend.utils import extract_handcrafted_features, extract_image_embedding


class Predictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.image_scaler = None
        self.image_pca = None
        self.original_feature_names = None
        self.image_model = None
        self.load_models()

    def load_models(self):
        """Загружаем модели один раз при старте"""
        print("Загрузка моделей для предсказания...")
        
        self.model137 = joblib.load("models/xgboost/xgb_tabular_handcrafted_images.pkl")
        self.model73 = joblib.load("models/xgboost/xgb_tabular_handcrafted.pkl")
        self.model = joblib.load("models/xgboost/xgb_tabular_only.pkl")
        self.image_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.preprocessor = joblib.load("models/preprocessor.pkl")
        self.image_scaler = joblib.load("models/image_scaler.pkl")
        self.image_pca = joblib.load("models/image_pca.pkl")
        self.original_feature_names = self.preprocessor.feature_names_in_
        
        print("Все модели успешно загружены")

    def _enrich_features(self, data: dict) -> pd.DataFrame:
        """Автоматически заполняет недостающие признаки"""
        df = pd.DataFrame([data])
        
        now = data.get("prediction_date") or datetime.now()
        
        # Временные признаки
        df['year'] = now.year
        month = now.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Производные признаки
        df['floor_ratio'] = df['floor'] / df['floors']
        
        # Близость к метро
        metros_close = ['Дубравная', 'Проспект Победы', 'Горки', 'Аметьево',
                       'Суконная слобода', 'Площадь Тукая', 'Козья Слобода',
                       'Кремлёвская', 'Яшьлек', 'Северный вокзал', 'Авиастроительная']
        df['is_close_to_metro'] = 1 if pd.notna(df['metro'].iloc[0]) and df['metro'].iloc[0] in metros_close else 0
        
        # avg_price_prev_month — временно ставим NaN (или можно улучшить позже)
        df['avg_price_prev_month'] = np.nan
        
        return df

    def predict(self, input_data: dict):
        start_time = time.time()
        
        # 1. Обогащаем табличные признаки
        tabular_df = self._enrich_features(input_data)[self.original_feature_names]
        # 2. Применяем preprocessor к табличным данным
        X_tabular = pd.DataFrame(self.preprocessor.transform(tabular_df), 
                                 columns=self.preprocessor.get_feature_names_out())
        
        # 3. Добавляем handcrafted признаки
        # print(X_tabular.shape)
        extra_features = []
        if input_data.get("description"):
            hc = extract_handcrafted_features(input_data["description"])
            extra_features.append(np.array([list(hc.values())]))
        
        # 4. Добавляем image embeddings
        if input_data.get("image_paths") and self.image_pca is not None and self.image_scaler is not None:
            emb = extract_image_embedding(self.image_model, input_data["image_paths"])
            if emb is not None:
                emb_scaled = self.image_scaler.transform([emb])
                emb_pca = self.image_pca.transform(emb_scaled)
                extra_features.append(emb_pca)
        
        # 5. Объединяем всё
        if extra_features:
            X_final = np.hstack([X_tabular] + extra_features)
            print(")0)")
        else:
            X_final = X_tabular
        
        # 6. Предсказание
        if X_final.shape[1] == 137:
            pred_price = float(self.model137.predict(X_final)[0])
            name = "XGBoost + Handcrafted + Images"
        elif X_final.shape[1] == 73:
            pred_price = float(self.model73.predict(X_final)[0])
            name = "XGBoost + Handcrafted"
        else:
            pred_price = float(self.model.predict(X_final)[0])
            name = "XGBoost"
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "recommended_price": round(pred_price),
            "lower_bound": round(pred_price * 0.88),
            "upper_bound": round(pred_price * 1.12),
            "confidence_interval": f"{round(pred_price * 0.88):,} — {round(pred_price * 1.12):,} ₽",
            "model_used": name,
            "processing_time_ms": round(processing_time, 1)
        }
