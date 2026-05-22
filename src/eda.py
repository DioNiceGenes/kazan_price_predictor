import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from .config import settings

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EDA:
    def __init__(self, tdf: pd.DataFrame):
        self.tdf = tdf.copy()
        self.figures_dir = Path(settings.ALL_DATA_PATH) / "reports" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"EDA: Данные загружены. Размер: {tdf.shape}")

    def save_plot(self, filename: str):
        path = self.figures_dir / filename
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {path}")
        plt.close()

    def run_full_eda(self):
        print(" Запуск полного EDA...")
        
        self.plot_target_distribution()
        self.plot_numerical_distributions()
        self.plot_categorical_distributions()
        self.plot_price_by_categories()
        self.plot_correlation_heatmap()
        self.plot_price_vs_key_features()
        self.plot_geographical_heatmap()
        self.plot_floor_and_ratio_analysis()
        self.plot_temporal_trends()
        
        print(f"\nEDA завершён. Все графики сохранены в {self.figures_dir}/")


    def plot_target_distribution(self):
        """Распределение целевой переменной"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.histplot(self.tdf['price'], bins=100, kde=True, ax=axes[0])
        axes[0].set_title('Распределение цены', fontsize=14)
        axes[0].set_xlabel('Цена (₽)')
        
        sns.histplot(np.log1p(self.tdf['price']), bins=100, kde=True, ax=axes[1])
        axes[1].set_title('Распределение log(Цена)', fontsize=14)
        axes[1].set_xlabel('log(Цена)')
        
        self.save_plot("01_price_distribution.png")

    def plot_numerical_distributions(self):
        """Распределения всех числовых признаков"""
        num_cols = self.tdf.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [col for col in num_cols if col not in ['lat', 'lng']]  # гео отдельно
        
        n = len(num_cols)
        cols = 3
        rows = (n + cols - 1) // cols
        
        plt.figure(figsize=(15, rows * 4))
        for i, col in enumerate(num_cols, 1):
            plt.subplot(rows, cols, i)
            sns.histplot(self.tdf[col], kde=True, bins=50)
            plt.title(f'Распределение {col}')
        
        plt.tight_layout()
        self.save_plot("02_numerical_distributions.png")

    def plot_categorical_distributions(self):
        """Распределения категориальных признаков"""
        cat_cols = ['rooms', 'is_studio', 'is_new_building', 'balcony', 'bathroom_type']
        cat_cols = [col for col in cat_cols if col in self.tdf.columns]
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cat_cols, 1):
            plt.subplot(2, 3, i)
            sns.countplot(data=self.tdf, x=col)
            plt.title(f'Распределение {col}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        self.save_plot("03_categorical_distributions.png")

    def plot_price_by_categories(self):
        """Цена в разрезе ключевых категорий"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.boxplot(x='rooms', y='price', data=self.tdf, ax=axes[0,0])
        axes[0,0].set_title('Цена по количеству комнат')
        
        sns.boxplot(x='is_new_building', y='price', data=self.tdf, ax=axes[0,1])
        axes[0,1].set_title('Цена: Новостройка vs Вторичка')
        
        sns.boxplot(x='balcony', y='price', data=self.tdf, ax=axes[1,0])
        axes[1,0].set_title('Цена в зависимости от балкона')
        
        sns.boxplot(x='bathroom_type', y='price', data=self.tdf, ax=axes[1,1])
        axes[1,1].set_title('Цена по типу санузла')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self.save_plot("04_price_by_categories.png")

    def plot_correlation_heatmap(self):
        numeric = self.tdf.select_dtypes(include=[np.number])
        corr = numeric.corr()
        
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Корреляционная матрица')
        self.save_plot("05_correlation_heatmap.png")

    def plot_price_vs_key_features(self):
        """Ключевые зависимости цены"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.scatterplot(x='total_square', y='price', hue='rooms', data=self.tdf, alpha=0.6, ax=axes[0,0])
        axes[0,0].set_title('Цена vs Общая площадь')
        
        sns.scatterplot(x='dist_to_center', y='price', data=self.tdf, alpha=0.6, ax=axes[0,1])
        axes[0,1].set_title('Цена vs Расстояние до центра')
        
        sns.scatterplot(x='dist_to_metro', y='price', data=self.tdf, alpha=0.6, ax=axes[1,0])
        axes[1,0].set_title('Цена vs Расстояние до метро')
        
        sns.boxplot(x='floor_ratio', y='price', data=self.tdf, ax=axes[1,1])
        axes[1,1].set_title('Цена vs Соотношение этажа')
        
        plt.tight_layout()
        self.save_plot("06_price_vs_key_features.png")

    def plot_geographical_heatmap(self):
        """Тепловая карта цен на карте Казани"""
        data = pd.read_csv(settings.RAW_DATA_PATH)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            data['coords_lng'], 
            data['coords_lat'], 
            c=data['price'], 
            cmap='viridis',
            alpha=0.6,
            s=8
        )
        plt.colorbar(scatter, label='Цена (₽)')
        plt.title('Географическое распределение цен в Казани')
        plt.xlabel('Долгота')
        plt.ylabel('Широта')
        self.save_plot("07_geographical_price_heatmap.png")

    def plot_floor_and_ratio_analysis(self):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x='floor', y='price', data=self.tdf[self.tdf['floor'] <= 20])
        plt.title('Цена по этажу')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='floor_ratio', y='price', data=self.tdf)
        plt.title('Цена по соотношению этажа к общему')
        self.save_plot("08_floor_analysis.png")

    def plot_temporal_trends(self):
        if 'time_source_updated' in self.tdf.columns:
            monthly_price = self.tdf.groupby(
                self.tdf['time_source_updated'].dt.to_period('M')
            )['price'].mean()
            
            plt.figure(figsize=(12, 6))
            monthly_price.plot(kind='line', marker='o')
            plt.title('Динамика средней цены по месяцам')
            plt.xlabel('Дата')
            plt.ylabel('Средняя цена (₽)')
            plt.grid(True)
            self.save_plot("09_temporal_price_trend.png")


if __name__ == "__main__":
    tdf = pd.read_csv(settings.TDF_PATH)
    eda = EDA(tdf)
    eda.run_full_eda()