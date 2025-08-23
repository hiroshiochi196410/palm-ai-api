# Dockerfile（Railway では Builder を Dockerfile に設定）
FROM node:20-bullseye-slim
WORKDIR /app

# 依存を先に入れてキャッシュ最適化
COPY package*.json ./
RUN npm ci --omit=dev

# アプリ本体（models/ を含む）をコピー
COPY . .

ENV NODE_ENV=production
ENV PORT=3000
EXPOSE 3000

CMD ["npm","start"]
