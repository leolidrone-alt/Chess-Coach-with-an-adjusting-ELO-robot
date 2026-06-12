import os
from openai import OpenAI
import chess.pgn
import io

class CloudReviewer:
    """极简云端复盘：使用 OpenAI 兼容接口调用 DeepSeek，返回分析文本。"""
    def __init__(self, api_key=None, model="deepseek-chat", ai_player_name="AI"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.ai_player_name = ai_player_name   # 新增
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

    def analyze(self, pgn_text: str) -> str:
        if not self.api_key:
            return "❌ 未配置 DEEPSEEK_API_KEY，无法进行云端复盘。"

        # 1. 解析 PGN，识别用户执子颜色
        user_color = "白方"  # 默认值
        try:
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)
            if game and game.headers:
                white_player = game.headers.get("White", "")
                black_player = game.headers.get("Black", "")
                if self.ai_player_name in white_player:
                    user_color = "黑方"
                elif self.ai_player_name in black_player:
                    user_color = "白方"
        except:
            pass  # 解析失败时使用默认值

        # 2. 构造 Prompt
        prompt = f"""你是一位国际象棋教练。请从**{user_color}**的角度分析下面这盘棋。
指出{user_color}的关键失误和亮点，用中文回复，语言通俗易懂，适合业余棋手阅读。
最后给出一个总体评价和下一步训练建议。

对局棋谱：
{pgn_text}"""

        # 3. 调用 API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业国际象棋教练，擅长分析业余棋手的对局。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ 云端复盘失败: {str(e)}"