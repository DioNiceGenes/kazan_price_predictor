from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.schemas import ApartmentInput, PredictionResponse, PricePrediction
from backend.predictor import Predictor
import os
import uuid
from pathlib import Path
from typing import List

app = FastAPI(title="Kazan Apartment Price API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация предиктора один раз при старте
predictor = Predictor()

# Директория для загруженных файлов
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(..., description="До 3 изображений")):
    """Загрузка изображений квартиры (до 3 шт)"""
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Максимум 3 изображения")
    
    if not files:
        raise HTTPException(status_code=400, detail="Необходимо загрузить хотя бы одно изображение")
    
    saved_files = []
    
    try:
        for file in files:
            # Проверка типа файла
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"Файл {file.filename} не является изображением")
            
            # Генерируем уникальное имя файла
            file_ext = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = UPLOAD_DIR / unique_filename
            
            # Сохраняем файл
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            
            saved_files.append({
                "filename": file.filename,
                "saved_path": str(file_path),
                "unique_id": unique_filename
            })
    
    except Exception as e:
        # Удаляем сохраненные файлы в случае ошибки
        for saved_file in saved_files:
            try:
                os.remove(saved_file["saved_path"])
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файлов: {str(e)}")
    
    return {
        "status": "success",
        "files": saved_files,
        "count": len(saved_files)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(input_data: ApartmentInput):
    try:
        result = predictor.predict(input_data.dict())
        return PredictionResponse(prediction=PricePrediction(**result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": predictor.model is not None}
