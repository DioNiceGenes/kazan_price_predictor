# Frontend Dockerfile
FROM node:20-alpine AS build

WORKDIR /app

# Копируем package.json и package-lock.json
COPY package*.json ./

# Установка зависимостей
RUN npm ci

# Копируем исходный код
COPY . .

# Сборка приложения
RUN npm run build

# Этап запуска - используем nginx для раздачи статических файлов
FROM nginx:alpine

# Копируем nginx конфиг
COPY nginx.conf /etc/nginx/nginx.conf

# Копируем собранное приложение из предыдущего этапа
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]
