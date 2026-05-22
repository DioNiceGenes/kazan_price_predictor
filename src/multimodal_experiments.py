import pandas as pd
from pathlib import Path

from config import settings
from preprocessing import create_features
from text_extraction import TextExperiments
from image_embeddings import ImageExperiments
from model_training import ModelTrainer


def run_final_experiments():
    print("="*100)
    print("ЗАПУСК ФИНАЛЬНЫХ ЭКСПЕРИМЕНТОВ")
    print("="*100)
    
    # ====================== ЗАГРУЗКА ДАННЫХ ======================
    tdf = pd.read_csv(settings.TDF_PATH)
    descriptions = pd.read_csv(settings.DESCRIPTIONS_PATH)['description']
    
    data_dict = create_features(tdf)
    
    X_raw = data_dict['X_raw']
    y = data_dict['y']
    
    # 1: Табличные + Handcrafted
    print("\n\n" + "★" * 30)
    print("ЭКСПЕРИМЕНТ 1: Табличные + Handcrafted Features")
    print("★" * 30)
    
    text_exp = TextExperiments(X_raw, y, descriptions)
    handcrafted = text_exp.create_handcrafted_features()
    
    X_hc = pd.concat([X_raw.reset_index(drop=True), 
                     handcrafted.reset_index(drop=True)], axis=1)
    
    trainer = ModelTrainer(X_raw, y, None, None)
    
    mae_hc, r2_hc, model_hc = trainer.evaluate_xgboost(
        X=X_hc,
        y=y,
        run_name="XGBoost_Tabular_Handcrafted",
        tags={"features": "tabular+handcrafted"}
    )

    # 2: Табличные + Handcrafted + Images
    print("\n\n" + "★" * 30)
    print("ЭКСПЕРИМЕНТ 2: Табличные + Handcrafted + Image Embeddings")
    print("★" * 30)
    
    image_exp = ImageExperiments(X_raw, y, tdf)
    image_pca = image_exp.extract_and_save_embeddings(n_components=64)
    
    image_df = pd.DataFrame(
        image_pca,
        columns=[f"img_pca_{i}" for i in range(64)],
        index=X_raw.index
    )
    
    X_full = pd.concat([
        X_hc.reset_index(drop=True), 
        image_df.reset_index(drop=True)
    ], axis=1)
    
    mae_full, r2_full, model_full = trainer.evaluate_xgboost(
        X=X_full,
        y=y,
        run_name="XGBoost_Tabular_Handcrafted_EfficientNet_PCA64",
        tags={"features": "tabular+handcrafted+image"}
    )

    results = {
        "Tabular Only": {"MAE": None, "R2": None},  # можно добавить baseline позже
        "Tabular + Handcrafted": {"MAE": mae_hc, "R2": r2_hc},
        "Tabular + Handcrafted + Images": {"MAE": mae_full, "R2": r2_full}
    }
    
    comparison = pd.DataFrame(results).T
    print("\n" + "="*100)
    print("ФИНАЛЬНОЕ СРАВНЕНИЕ")
    print("="*100)
    print(comparison.round(4))
    
    comparison.to_csv("reports/final_experiments_comparison.csv")
    
    print(f"\nЭксперименты завершены! Результаты сохранены в reports/")
    return comparison


if __name__ == "__main__":
    run_final_experiments()