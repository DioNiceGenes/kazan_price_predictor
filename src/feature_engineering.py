import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from config import settings


KZN_CENTER_LAT = 55.7887
KZN_CENTER_LON = 49.1221

metros_close = ['Дубравная', 'Проспект Победы', 'Горки', 'Аметьево', 
                'Суконная слобода', 'Площадь Тукая', 'Козья Слобода', 
                'Кремлёвская', 'Яшьлек', 'Северный вокзал', 'Авиастроительная']


def haversine_distance(lat, lon, lat2=KZN_CENTER_LAT, lon2=KZN_CENTER_LON):
    """Расстояние до центра Казани"""
    R = 6371.0
    lat1_rad = np.radians(lat)
    lon1_rad = np.radians(lon)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def load_raw_data() -> pd.DataFrame:
    print("Загрузка raw данных...")
    data = pd.read_csv(settings.RAW_DATA_PATH)
    print(f"Загружено {len(data)} записей")
    return data


def create_image_folders(data: pd.DataFrame) -> pd.Series:
    print("Создание путей к папкам с изображениями...")
    
    def make_folder_path(row):
        try:
            dt = datetime.strptime(row['time_source_updated'], "%Y-%m-%d %H:%M:%S")
            return f"{settings.IMAGES_BASE_PATH}/{row['Unnamed: 0']}_{dt.strftime('%Y-%m-%d%H%M%S')}"
        except:
            return None
    
    data['image_folder'] = data.apply(make_folder_path, axis=1)
    return data['image_folder']


def clean_and_create_tdf(data: pd.DataFrame) -> pd.DataFrame:
    """Полная очистка + создание tdf + пространственно-временные признаки"""
    print("Очистка и создание tdf...")
    
    df = data.copy()
    
    # 1. Удаление ненужных колонок
    non_table_cols = [
        'params2_О доме_Этажей в доме', 'params2_О квартире_Этаж',
        'params2_О квартире_Жилая площадь', 'params2_О квартире_Количество комнат',
        'params2_О квартире_Общая площадь', 'params2_О квартире_Площадь кухни',
        'metro_only', 'district_only', 'address', 'Unnamed: 0',
        'param_1945', 'param_1957', 'param_2009',
        'params2_О доме_Год постройки', 'params2_О доме_Грузовой лифт',
        'params2_О доме_Двор', 'params2_О доме_Парковка',
        'params2_О доме_Пассажирский лифт', 'params2_О квартире_Высота потолков, м',
        'params2_О квартире_Тип комнат', 'params_Тип дома',
        'params2_О квартире_Вид сделки', 'params2_О квартире_Статус',
        'params2_О квартире_Ремонт', 'title', 'description', 'image_folder', 'images',
        'params2_Об объекте'
    ]
    
    tdf = df.drop(columns=[col for col in non_table_cols if col in df.columns], errors='ignore')
    
    # 2. Переименование колонок
    new_names = {
        "coords_lat": "lat", "coords_lng": "lng", "km_do_metro": "dist_to_metro",
        'params2_О квартире_Балкон или лоджия': "balcony",
        'params2_О квартире_Санузел': "bathroom_type",
        'params2_О квартире_Способ продажи': "sell_type",
        'params_Этажей в доме': "floors", 'params_Этаж': "floor",
        'params_Жилая площадь': "living_square",
        'params_Количество комнат': "rooms",
        'params_Площадь': "total_square",
        'params_Площадь кухни': "kitchen_square",
        'params_Вид объекта': "is_new_building",
    }
    tdf.rename(columns=new_names, inplace=True)
    
    # 3. Обработка rooms и studio
    tdf["is_studio"] = tdf["rooms"].apply(lambda x: 1 if x == 'Студия' or pd.isna(x) else 0)
    tdf["rooms"] = tdf["rooms"].apply(lambda x: int(x) if pd.notna(x) and x != "Студия" and str(x).isdigit() else 0)
    
    # 4. Приведение строк к нижнему регистру
    for col in ['balcony', 'bathroom_type', 'sell_type']:
        if col in tdf.columns:
            tdf[col] = tdf[col].apply(lambda x: str(x).lower() if pd.notna(x) else x)
    
    tdf["time_source_updated"] = pd.to_datetime(tdf['time_source_updated'])
    
    # 5. Заполнение пропусков
    tdf['balcony'] = tdf['balcony'].fillna('нет')
    tdf['bathroom_type'] = tdf['bathroom_type'].fillna(tdf['bathroom_type'].mode()[0])
    tdf["sell_type"] = tdf["sell_type"].fillna(tdf["sell_type"].mode()[0])
    
    tdf['living_square'] = tdf.groupby('rooms')['living_square'].transform(lambda x: x.fillna(x.median()))
    tdf['living_square'] = tdf['living_square'].fillna(tdf['living_square'].median())
    tdf['kitchen_square'] = tdf.groupby('rooms')['kitchen_square'].transform(lambda x: x.fillna(x.median()))
    
    # 6. Пространственно-временные признаки
    print("Создание пространственно-временных признаков...")
    
    # Расстояние до центра
    tdf['dist_to_center'] = tdf.apply(
        lambda row: haversine_distance(row['lat'], row['lng']) 
        if pd.notna(row['lat']) and pd.notna(row['lng']) else np.nan, axis=1
    )
    
    # Близость к метро
    tdf["is_close_to_metro"] = tdf["metro"].apply(
        lambda x: 1 if pd.notna(x) and x in metros_close else 0
    )
    
    # Временные признаки
    tdf['year'] = tdf['time_source_updated'].dt.year
    tdf['month_sin'] = np.sin(2 * np.pi * tdf['time_source_updated'].dt.month / 12)
    tdf['month_cos'] = np.cos(2 * np.pi * tdf['time_source_updated'].dt.month / 12)
    tdf['avg_price_prev_month'] = tdf['price'].shift(1).rolling(30).mean()
    tdf = tdf.dropna(subset=["avg_price_prev_month"])
    tdf = tdf.drop(["time_source_updated", "lat", "lng"], axis=1)
    
    # Соотношение этажей
    tdf['floor_ratio'] = tdf['floor'] / tdf['floors']
    tdf["is_new_building"] = tdf["is_new_building"].fillna(tdf["is_new_building"].mode()[0])
    tdf["is_new_building"] = tdf["is_new_building"].apply(lambda x: int(x == "Новостройка"))

    # Выбросы
    cols_woutliers = ('living_square', 'kitchen_square', 'total_square', 'floors', 'price')
    normal_conditions = True

    for col in cols_woutliers:
        Q1 = tdf[col].quantile(0.1)
        Q3 = tdf[col].quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        normal_conditions &= (tdf[col] >= lower_bound) & (tdf[col] <= upper_bound)
    tdf = pd.DataFrame(tdf.loc[normal_conditions], index=tdf.loc[normal_conditions].index)
    print(f"tdf создан. Финальный размер: {tdf.shape}")
    return tdf


def save_processed_data(tdf: pd.DataFrame, image_folders: pd.Series, descriptions: pd.Series):
    """Сохранение всех обработанных данных"""
    Path(settings.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    
    tdf.to_csv(settings.TDF_PATH, index=False)
    image_folders.to_csv(settings.IMAGE_FOLDERS_PATH, index=False)
    descriptions.to_csv(settings.DESCRIPTIONS_PATH, index=False)
    
    print(f"  Данные успешно сохранены в {settings.PROCESSED_DIR}/")
    print(f"   - tdf: {tdf.shape}")
    print(f"   - image_folders: {len(image_folders)}")
    print(f"   - descriptions: {len(descriptions)}")


if __name__ == "__main__":
    data = load_raw_data()
    tdf = clean_and_create_tdf(data)
    image_folders = create_image_folders(data.loc[tdf.index])
    descriptions = data.loc[tdf.index, 'description']
    save_processed_data(tdf, image_folders, descriptions)