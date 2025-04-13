FROM python:3.9-slim

WORKDIR /app

# 必要なパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 起動時のコマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 