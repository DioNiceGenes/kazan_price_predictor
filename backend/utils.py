import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path


def extract_handcrafted_features(description: str) -> dict:
    """Извлекает hand-crafted признаки из описания"""
    if not description or len(description.strip()) == 0:
        return {k: 0 for k in [
            'desc_len_chars', 'desc_len_words', 'excl_marks', 'question_marks',
            'has_euroremont', 'has_designer', 'has_cosmetic', 'has_needs_repair',
            'has_new_building', 'has_furniture', 'has_view', 'has_river_view',
            'has_balcony', 'has_loggia', 'has_wardrobe', 'mentions_kitchen',
            'mentions_bathroom'
        ]}
    
    t = str(description).lower()
    return {
        'desc_len_chars': len(description),
        'desc_len_words': len(description.split()),
        'excl_marks': description.count('!'),
        'question_marks': description.count('?'),
        'has_euroremont': int(any(x in t for x in ['евроремонт', 'евро ремонт', 'евро-ремонт'])),
        'has_designer': int(any(x in t for x in ['дизайнерский', 'авторский'])),
        'has_cosmetic': int('косметический' in t),
        'has_needs_repair': int(any(x in t for x in ['требует ремонта', 'нуждается в ремонте'])),
        'has_new_building': int(any(x in t for x in ['новостройк', 'новый дом', 'сдан'])),
        'has_furniture': int(any(x in t for x in ['мебель', 'меблирован', 'с мебелью'])),
        'has_view': int('вид на' in t),
        'has_river_view': int(any(x in t for x in ['вид на реку', 'вид на волгу', 'вид на казанку'])),
        'has_balcony': int('балкон' in t),
        'has_loggia': int('лоджи' in t),
        'has_wardrobe': int(any(x in t for x in ['шкаф', 'гардероб'])),
        'mentions_kitchen': int(any(x in t for x in ['кухн', 'кухон'])),
        'mentions_bathroom': int(any(x in t for x in ['санузел', 'ванн', 'туалет'])),
    }


def extract_image_embedding(model, image_paths: list, max_images=3):
    """Извлекает средний эмбеддинг из списка фотографий"""
    if not image_paths:
        return None
    
    # Загружаем модель один раз (можно вынести в глобальную переменную)
    try:
        model.eval()
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        embeddings = []
        for path in image_paths[:max_images]:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    emb = feature_extractor(img_tensor)
                    emb = torch.flatten(emb, 1).numpy().squeeze()
                embeddings.append(emb)
            except:
                continue
                
        if len(embeddings) == 0:
            return None
        return np.mean(embeddings, axis=0)
        
    except Exception as e:
        print(f"Ошибка извлечения эмбеддинга: {e}")
        return None
