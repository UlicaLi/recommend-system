import os
import logging
from typing import Optional

class Settings:
    # --- Database Configuration ---
    # 建议通过环境变量注入: export DB_URL="mysql+pymysql://user:pass@host:3306/dbname"
    DB_URL: str = os.getenv("DB_URL", "mysql+pymysql://root:mysqlroot@127.0.0.1:3306/rec_sys")
    
    # --- Redis Configuration ---
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_KEY_PREFIX: str = "rec:sys:"
    REDIS_EXPIRE_SECONDS: int = 86400 # 推荐结果缓存 24 小时

    # --- Recommendation Business Rules ---
    REC_HISTORY_COUNT: int = 4     # 场景A: 常用工具展示数
    REC_DISCOVERY_COUNT: int = 8   # 场景B: 猜你喜欢展示数
    REC_RELATED_COUNT: int = 5     # 相关推荐展示数
    
    # --- Data Preprocessing ---
    TIME_DECAY_WINDOW: int = 180   # 仅计算最近 180 天的数据
    TIME_DECAY_RATE: float = 0.95  # 时间衰减系数 (每天衰减 5%)
    min_interaction_threshold: int = 3 # 过滤掉交互极少的噪音数据(可选)

    # --- ALS Model Hyperparameters ---
    ALS_FACTORS: int = 64          # 隐向量维度
    ALS_REGULARIZATION: float = 0.05
    ALS_ITERATIONS: int = 20
    ALS_ALPHA: float = 40.0        # 置信度缩放系数
    
    # --- System ---
    LOG_LEVEL: int = logging.INFO
    LOG_DIR: str = "logs"

settings = Settings()

# 简单的日志配置
os.makedirs(settings.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"{settings.LOG_DIR}/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RecSys")