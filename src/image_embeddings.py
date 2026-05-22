import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

from config import settings
from model_training import ModelTrainer
from preprocessing import create_features


class ImageExperiments:
    def __init__(self, X_raw: pd.DataFrame, y: pd.Series, data_with_images: pd.DataFrame):
        """
        X_raw — сырые табличные признаки (до preprocessor)
        y — цены
        data_with_images — датафрейм, в котором есть колонка 'image_folder'
        """
        self.X_raw = X_raw.copy()
        self.y = y.copy()
        self.data = data_with_images.copy()
        
        # Выравниваем
        common_idx = X_raw.index
        self.data = self.data.loc[common_idx]
        self.image_folders = self.data['image_folder']
        
        print(f"ImageExperiments инициализирован:")
        print(f"   X_raw.shape = {self.X_raw.shape}")
        print(f"   Кол-во папок с фото = {self.image_folders.notna().sum()}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

    def load_efficientnet(self):
        """Загружаем EfficientNet-B4"""
        print("Загрузка EfficientNet-B4...")
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Убираем классификатор
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        print("EfficientNet-B4 загружен на", self.device)

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Извлекает эмбеддинг из одного изображения"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.feature_extractor(img)
                emb = torch.flatten(emb, 1).cpu().numpy().squeeze()
            return emb
        except:
            return None

    def get_mean_embedding(self, folder_path: str) -> np.ndarray:
        """Средний эмбеддинг по всем фото в папке (до 3 фото)"""
        if pd.isna(folder_path) or not os.path.exists(folder_path):
            return np.zeros(1792)  # размерность B4
        
        embeddings = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder_path, filename)
                emb = self.extract_embedding(path)
                if emb is not None:
                    embeddings.append(emb)
                if len(embeddings) >= 3:  # берем максимум 3 фото
                    break
        
        if len(embeddings) == 0:
            return np.zeros(1792)
        
        return np.mean(embeddings, axis=0)

    def extract_all_image_embeddings(self, save_path="models/image_embeddings.npy"):
        """Извлекает эмбеддинги для всех объявлений"""

        file_path = "models/image_embeddings.npy"
        if Path(file_path).exists():
            embeddings = np.load(file_path)
            print("Эмбеддинги уже загружены!")
            return embeddings

        print("Извлечение эмбеддингов из изображений...")
        self.load_efficientnet()
        
        embeddings = []
        for folder in tqdm(self.image_folders, desc="Processing images"):
            emb = self.get_mean_embedding(folder)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        np.save(save_path, embeddings)
        print(f"Эмбеддинги сохранены: {embeddings.shape}")
        
        return embeddings

    def run_experiment(self, n_components=64):
        """Полный эксперимент: EfficientNet + табличные признаки"""
        print("\n" + "="*90)
        print("ЭКСПЕРИМЕНТ: EfficientNet-B4 Image Embeddings")
        print("="*90)
        
        # Извлекаем эмбеддинги
        embeddings = self.extract_all_image_embeddings()
        
        # PCA + Scaling
        print(f"Применяем PCA ({n_components} компонент)...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        
        print(f"Доля объяснённой дисперсии: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Объединяем с табличными признаками
        embeddings_df = pd.DataFrame(embeddings_pca, 
                                   columns=[f"img_pca_{i}" for i in range(n_components)],
                                   index=self.X_raw.index)
        embeddings_df.to_csv(settings.PROCESSED_DIR + "/image_embeddings.csv")
        
        X_with_img = pd.concat([self.X_raw.reset_index(drop=True), 
                               embeddings_df.reset_index(drop=True)], axis=1)
        
        # Запускаем XGBoost
        trainer = ModelTrainer(self.X_raw, self.y, None, None)  # временно
        
        mae_img, r2_img, model = trainer.evaluate_xgboost(
            X=X_with_img,
            y=self.y,
            run_name=f"XGBoost_EfficientNet_B4_PCA{n_components}",
            tags={"modality": "image", "pca": n_components}
        )
        
        # Сохраняем артефакты
        joblib.dump(scaler, "models/image_scaler.pkl")
        joblib.dump(pca, "models/image_pca.pkl")
        
        return mae_img, r2_img, model


if __name__ == "__main__":
    tdf = pd.read_csv(settings.TDF_PATH)
    data_dict = create_features(tdf)
    image_folders = pd.read_csv(settings.IMAGE_FOLDERS_PATH)

    preprocessor = data_dict["preprocessor"]
    X = pd.DataFrame(preprocessor.transform(data_dict['X']), columns=data_dict['feature_names'], index=data_dict["X"].index)
    y = data_dict["y"]
    
    experiments = ImageExperiments(
        X_raw=X,
        y=y,
        data_with_images=image_folders
    )
    
    mae, r2, model = experiments.run_experiment(n_components=64)