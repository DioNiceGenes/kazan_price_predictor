from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ApartmentInput(BaseModel):
    # Обязательные / основные параметры
    total_square: float = Field(..., gt=0, description="Общая площадь")
    rooms: int = Field(..., ge=0, description="Количество комнат (0 для студии)")
    floor: int = Field(..., ge=1)
    floors: int = Field(..., ge=1)
    
    # Опциональные табличные параметры
    living_square: Optional[float] = None
    kitchen_square: Optional[float] = None
    dist_to_metro: Optional[float] = None
    dist_to_center: Optional[float] = None
    balcony: str = "нет"
    bathroom_type: str = "совмещенный"
    sell_type: str = "свободная продажа"
    metro: Optional[str] = None
    is_new_building: bool = False
    is_studio: bool = False
    
    # Дополнительные данные
    description: Optional[str] = Field(None, description="Описание объявления")
    image_paths: Optional[List[str]] = Field(None, description="Список путей к фотографиям (до 3 шт)")
    prediction_date: Optional[datetime] = None
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "total_square": 58.5,
                "rooms": 2,
                "floor": 9,
                "floors": 16,
                "living_square": 32.0,
                "kitchen_square": 9.5,
                "dist_to_metro": 0.65,
                "dist_to_center": 2.8,
                "balcony": "есть",
                "bathroom_type": "совмещенный",
                "sell_type": "свободная продажа",
                "metro": "Кремлёвская",
                "is_new_building": False,
                "is_studio": False,
                "description": "Просторная квартира в хорошем состоянии. Евроремонт, новая сантехника, встроенная кухня. Вид на парк. Рядом вся инфраструктура.",
                "image_paths": [
                    "/home/bakamol/Desktop/code/WebProjects/kazan_price_predictor/data/images/0_2023-10-291026411/1.jpg",
                    "/home/bakamol/Desktop/code/WebProjects/kazan_price_predictor/data/images/0_2023-10-29102641/2.jpg",
                    "/home/bakamol/Desktop/code/WebProjects/kazan_price_predictor/data/images/0_2023-10-29102641/3.jpg",
                ],
                "prediction_date": "2026-05-12T12:00:00"
            }
        }


class PricePrediction(BaseModel):
    recommended_price: int
    lower_bound: int
    upper_bound: int
    confidence_interval: str
    model_used: str
    processing_time_ms: float


class PredictionResponse(BaseModel):
    status: str = "success"
    prediction: PricePrediction
