# Dockerfile
FROM node:18-bullseye-slim

# 作業ディレクトリ
WORKDIR /app

# 余計な出力を減らしつつ、prod だけ入れる設定
ENV NODE_ENV=production \
    npm_config_loglevel=warn

# 依存インストール（package-lock があれば npm ci、無ければ npm install）
COPY package*.json ./
RUN set -eux; \
    if [ -f package-lock.json ]; then \
      npm ci --omit=dev --no-audit --no-fund; \
    else \
      npm install --omit=dev --no-audit --no-fund; \
    fi

# アプリ本体
COPY server.js ./

# 🔴 モデルを明示的にコピー（これが重要）
COPY models/model.json models/model.json
COPY models/group1-shard*.bin models/

# 動作確認用に中身を一度表示（任意）
RUN ls -lh models

# ポート
ENV PORT=8080
EXPOSE 8080

# 起動
CMD ["node", "server.js"]
