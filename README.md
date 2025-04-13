# Python Server Examples

このリポジトリには2つのPythonサーバーの例が含まれています。

## 1. シンプルなFastAPIサーバー（simple_server.py）

商品管理のための基本的なRESTful APIサーバーです。LLMモデルは不要で、すぐに起動できます。

### 起動方法

```bash
python3 -m uvicorn simple_server:app --host 0.0.0.0 --port 8000
```

### 使用方法

- `GET /` - ウェルカムメッセージを表示
- `GET /items/` - すべての商品を取得
- `GET /items/{item_id}` - 特定の商品を取得
- `POST /items/` - 新しい商品を追加
- `PUT /items/{item_id}` - 既存の商品を更新
- `DELETE /items/{item_id}` - 商品を削除

APIドキュメントは http://localhost:8000/docs で確認できます。

## 2. LLMを使用したAPIサーバー（main.py）

LLaMA 2ベースのELYZA日本語モデルを使用したチャットAPIサーバーです。

### 必要条件

- モデルファイルをダウンロードし、`models`ディレクトリに配置する必要があります。
- `main.py`内の`MODEL_PATH`がモデルファイルの正しいパスを指している必要があります。

### 起動方法

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 使用方法

- `GET /` - ウェルカムメッセージを表示
- `POST /v1/chat/completions` - チャット完了APIを使用

リクエスト例:
```json
{
  "messages": [
    {"role": "system", "content": "あなたは親切なアシスタントです。"},
    {"role": "user", "content": "こんにちは！"}
  ],
  "max_tokens": 150,
  "temperature": 0.7
}
```

## 依存関係のインストール

```bash
python3 -m pip install fastapi uvicorn llama-cpp-python
```
