# main.py
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import os
import json
import re
import logging

# --- ロガーの設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 定数定義 ---
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

# グローバル変数
tokenizer = None
model = None
DEVICE = "cpu"

# --- FastAPIアプリケーションの初期化 ---
app = FastAPI(
    title="Communication Analysis API",
    description="API to analyze communication text and prevent misunderstandings using ELYZA model.",
    version="1.1.0",
)

# --- Pydanticモデル定義 ---
class ContextInfo(BaseModel):
    """オプションのコンテキスト情報"""
    audience_relationship: Optional[Literal['上司', '同僚', '部下', '顧客', '友人', '不明']] = Field(
        None, description="相手との関係性"
    )
    communication_purpose: Optional[Literal['依頼', '報告', '質問', '謝罪', '提案', '情報共有', '不明']] = Field(
        None, description="コミュニケーションの目的"
    )
    previous_context: Optional[str] = Field(
        None, description="直前の会話履歴や関連情報など、文脈理解の助けとなるテキスト"
    )

class AnalyzeRequest(BaseModel):
    """/analyze_communication へのリクエストボディ"""
    text: str = Field(..., description="分析対象の文章")
    context: Optional[ContextInfo] = Field(None, description="オプションのコンテキスト情報")

class AnalysisDetail(BaseModel):
    """分析結果の詳細"""
    risk_type: str = Field(..., description="検出されたリスクの種類 (例: '曖昧さ(指示)')")
    problematic_segment: Optional[str] = Field(None, description="問題があると判断された具体的な文章箇所")
    explanation: str = Field(..., description="なぜ問題なのか、どのような齟齬を生むかの説明")
    suggestion: Optional[str] = Field(None, description="具体的な改善案や代替表現")

class AnalyzeResponse(BaseModel):
    """/analyze_communication のレスポンスボディ"""
    original_text: str = Field(..., description="入力された元の文章")
    overall_risk_level: Literal['High', 'Medium', 'Low', 'None', 'Unknown'] = Field(
        ..., description="総合的な齟齬リスク評価"
    )
    analysis_details: List[AnalysisDetail] = Field(..., description="分析結果の詳細リスト")
    improved_example: Optional[str] = Field(None, description="全体を改善した場合の例文 (任意)")

def load_model():
    """モデルをオンデマンドでロードする関数"""
    global tokenizer, model, DEVICE
    
    if tokenizer is not None and model is not None:
        return True
        
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading model: {MODEL_NAME}...")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

        if torch.cuda.is_available():
            DEVICE = "cuda"
            model = model.to(DEVICE)
            logger.info(f"Model loaded on GPU ({DEVICE})")
        else:
            DEVICE = "cpu"
            logger.warning("CUDA not available. Model loaded on CPU. This will be very slow!")
            
        return True
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
        return False

# --- APIエンドポイント ---
@app.post("/analyze_communication", response_model=AnalyzeResponse)
async def analyze_communication(request: AnalyzeRequest):
    """コミュニケーションテキストを分析し、齟齬が生じる可能性のある部分を特定します。"""
    
    # モデルのロード
    if not load_model():
        raise HTTPException(
            status_code=503,
            detail="モデルのロードに失敗しました。サーバー管理者に連絡してください。"
        )
    
    # システムプロンプト（モデルの動作指示）
    system_prompt = """あなたは日本語のコミュニケーションを分析する専門家です。
    与えられた文章を解析し、齟齬や誤解を生む可能性のある部分を特定してください。
    
    # 分析すべき項目
    - 曖昧な表現（5W1Hが不明確、指示が不明瞭など相手の解釈にズレが生じる可能性）
    
    # 出力形式
    必ずJSON形式で、以下の構造に従って出力してください：
    
    ```json
    {
      "overall_risk_level": "High/Medium/Low/None",
      "analysis_details": [
        {
          "risk_type": "リスクの種類 (例: '曖昧さ(指示)')",
          "problematic_segment": "問題がある具体的な文章箇所",
          "explanation": "なぜ問題か、どのような齟齬を生むかの説明",
          "suggestion": "改善案"
        },
        // 複数のリスクがあれば追加
      ],
      "improved_example": "全体を改善した文例"
    }
    ```
    """
    
    # --- ユーザー入力部分のプロンプト構築 ---
    context_text = ""
    if request.context:
        context_text += "\n\n# コンテキスト情報:\n"
        if request.context.audience_relationship:
            context_text += f"- 相手との関係性: {request.context.audience_relationship}\n"
        if request.context.communication_purpose:
            context_text += f"- コミュニケーションの目的: {request.context.communication_purpose}\n"
        if request.context.previous_context:
            context_text += f"- 事前情報/会話履歴:\n{request.context.previous_context}\n"

    user_prompt = f"""以下の文章を、提供されたコンテキスト情報を踏まえて徹底的に分析し、指示されたJSON形式で結果を出力してください。

    分析対象文章:
    {request.text}
    {context_text}
    """
    # --- モデルへの入力プロンプト全体 ---
    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{system_prompt}{E_SYS}",
        prompt=user_prompt,
        e_inst=E_INST,
    )

    try:
        with torch.no_grad():
            token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            logger.info("Analyzing communication risk...")
            # 注意: 長いプロンプトの場合、モデルの最大コンテキスト長を超える可能性がある
            # max_new_tokens も応答JSONの長さを考慮して設定
            output_ids = model.generate(
                token_ids.to(DEVICE),
                max_new_tokens=1024, # JSON応答用に十分な長さを確保 (必要なら増やす)
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.1, # 安定したJSON出力を期待して低めに設定
                top_p=0.95,      # 念のため設定
                do_sample=False, # 決定的な出力を得るためにFalse推奨
            )
            logger.info("Analysis complete.")

        # --- 出力 (JSON文字列を期待) のデコードとパース ---
        output_text = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        logger.info(f"Raw model output:\n{output_text}") # デバッグ用に生の出力を表示

        # --- JSONパース処理 ---
        try:
            # モデル出力からJSON部分を抽出 (マークダウン形式を想定)
            json_str = output_text # デフォルトはそのまま使う
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # マークダウンがない場合、出力全体がJSONであると仮定するか、
                # {} で囲まれた部分を探すなどの処理を追加することもできる
                logger.warning("Could not find JSON within Markdown code block. Trying to parse the whole output.")
                # 念のため、応答の前後に不要なテキストがあれば取り除く試み
                first_brace = json_str.find('{')
                last_brace = json_str.rfind('}')
                if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                    json_str = json_str[first_brace:last_brace+1]


            parsed_data = json.loads(json_str)

            # Pydanticモデルでバリデーションしつつレスポンス作成
            # analysis_details がリストであることを確認
            details_data = parsed_data.get("analysis_details", [])
            if not isinstance(details_data, list):
                logger.error(f"analysis_details is not a list in model output: {details_data}")
                details_data = [] # エラーの場合は空にする

            # 各詳細オブジェクトが必要なキーを持つか確認しつつ変換
            valid_details = []
            for detail in details_data:
                if isinstance(detail, dict) and "risk_type" in detail and "explanation" in detail:
                     # オプショナルなキーは .get() で安全にアクセス
                    valid_details.append(
                        AnalysisDetail(
                            risk_type=detail["risk_type"],
                            problematic_segment=detail.get("problematic_segment"),
                            explanation=detail["explanation"],
                            suggestion=detail.get("suggestion")
                        )
                    )
                else:
                     logger.warning(f"Skipping invalid analysis detail item: {detail}")


            response_data = AnalyzeResponse(
                original_text=request.text,
                overall_risk_level=parsed_data.get("overall_risk_level", "Unknown"),
                analysis_details=valid_details,
                improved_example=parsed_data.get("improved_example")
            )
            logger.info("Successfully parsed model output and created response.")
            return response_data

        except (json.JSONDecodeError, TypeError, KeyError, Exception) as json_e:
            logger.error(f"Error parsing model output as JSON: {json_e}", exc_info=True)
            # JSONパース失敗時のフォールバック処理
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse model output as valid JSON. Please check the model's response format. Raw output: {output_text}"
            )

    except Exception as e:
        logger.error(f"Error during communication analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during communication analysis.")

# --- ヘルスチェック用エンドポイント (任意) ---
@app.get("/health")
async def health_check():
    """Returns the health status of the API."""
    model_loaded = tokenizer is not None and model is not None
    return {
        "status": "ok", 
        "model_name": MODEL_NAME, 
        "model_loaded": model_loaded,
        "device": DEVICE if model_loaded else "not loaded"
    }

@app.get("/")
async def root():
    """API のルートエンドポイント。"""
    return {
        "message": "Communication Analysis API",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health - Check API status",
            "analyze": "/analyze_communication - Analyze communication text"
        }
    }

# --- Uvicornで実行するためのコード (直接実行する場合) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    # 本番環境では --reload は外す
    # host="0.0.0.0" は外部からのアクセスを許可 (セキュリティに注意)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


