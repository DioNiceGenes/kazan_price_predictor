# Docker запуск приложения

## Требования
- Docker
- Docker Compose

## Структура Docker

### Файлы
- `Dockerfile` - для backend (Python/FastAPI)
- `frontend/Dockerfile` - для frontend production сборки (Node + Nginx)
- `frontend/Dockerfile.dev` - для frontend разработки (Node dev сервер)
- `frontend/nginx.conf` - конфигурация Nginx для frontend
- `docker-compose.yml` - production конфигурация
- `docker-compose.dev.yml` - development конфигурация

## Запуск в Production

```bash
docker-compose up --build
```

Приложение будет доступно по адресу:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

## Запуск в Development

```bash
docker-compose -f docker-compose.dev.yml up --build
```

Преимущества dev режима:
- Frontend работает на dev сервере Vite (с hot reload)
- Backend переиспользует код из папки (можно редактировать в реальном времени)
- Логирование выводится в консоль

## Останавливаем контейнеры

```bash
# Production
docker-compose down

# Development
docker-compose -f docker-compose.dev.yml down
```

## Удаление всех данных и контейнеров

```bash
docker-compose down -v
```

## Просмотр логов

```bash
# Все сервисы
docker-compose logs -f

# Только backend
docker-compose logs -f backend

# Только frontend
docker-compose logs -f frontend
```

## Сборка образов вручную

```bash
# Backend
docker build -t kazan-price-predictor-backend .

# Frontend production
docker build -t kazan-price-predictor-frontend ./frontend -f frontend/Dockerfile

# Frontend development
docker build -t kazan-price-predictor-frontend-dev ./frontend -f frontend/Dockerfile.dev
```

## Запуск контейнеров вручную

```bash
# Backend
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  kazan-price-predictor-backend

# Frontend (production)
docker run -p 5173:5173 kazan-price-predictor-frontend

# Frontend (development)
docker run -p 5173:5173 \
  -v $(pwd)/frontend/src:/app/src \
  kazan-price-predictor-frontend-dev
```

## Переменные окружения

### Backend
- `PYTHONUNBUFFERED=1` - вывод логов в реальном времени

### Frontend
- `VITE_API_URL` - URL API (в dev: http://localhost:8000)

## Общие порты

| Сервис  | Порт | URL                    |
|---------|------|----------------------|
| Frontend| 5173 | http://localhost:5173 |
| Backend | 8000 | http://localhost:8000 |

## Health Check

Backend имеет встроенный health check:
```bash
curl http://localhost:8000/health
```

## Проблемы и решения

### Port already in use
Если порт уже занят, измените в docker-compose.yml:
```yaml
ports:
  - "8001:8000"  # новый порт хоста:порт контейнера
```

### No space left on device
Очистите неиспользуемые образы:
```bash
docker system prune -a
```

### Permission denied
На Linux может потребоваться sudo:
```bash
sudo docker-compose up
```

## Монтирование папок

В docker-compose используются volumes для:
- `./data` - сохраненные загруженные фото
- `./models` - модели ML (для быстрого доступа)
- `./backend` и `./src` - исходный код (в dev режиме)
