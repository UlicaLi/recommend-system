from contextlib import asynccontextmanager
from typing import List

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Path
from config.settings import settings

# 全局 Redis 连接池
redis_client: redis.Redis = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理：
    启动时初始化 Redis 连接池
    关闭时断开连接
    """
    global redis_client
    # 使用 settings 中的配置
    redis_client = redis.from_url(
        settings.REDIS_URL, 
        decode_responses=True, # 自动解码为字符串
        encoding="utf-8"
    )
    try:
        await redis_client.ping()
        print(f"Connected to Redis at {settings.REDIS_URL}")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
    
    yield
    
    if redis_client:
        await redis_client.close()
        print("Redis connection closed")

app = FastAPI(
    title="Tool RecSys API",
    description="提供工具推荐服务的 RESTful 接口",
    version="0.1.0",
    lifespan=lifespan
)

async def get_list_from_redis(key: str) -> List[int]:
    """通用辅助函数：从 Redis 获取 List 并转为 int 列表"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis service unavailable")
    
    try:
        # lrange key 0 -1 获取整个列表
        items = await redis_client.lrange(key, 0, -1)
        # Redis 存的是字符串，转回 int
        return [int(x) for x in items]
    except redis.RedisError as e:
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")
    except ValueError:
        # 如果数据损坏无法转 int，返回空或报错，这里选择过滤
        return []

# --- 接口定义 ---

@app.get("/recommend/history/{user_id}", response_model=List[int], tags=["Recommendations"])
async def get_user_history_recommendations(user_id: int = Path(..., title="用户ID", gt=0)):
    """
    获取用户的【常用工具】推荐 (基于历史频率)
    若用户无历史，返回全局热门
    """
    key = f"{settings.REDIS_KEY_PREFIX}user:{user_id}:history"
    recs = await get_list_from_redis(key)
    if not recs:
        # 冷启动：无历史则返回全局热门
        recs = await get_list_from_redis(f"{settings.REDIS_KEY_PREFIX}global:popular")
    return recs

@app.get("/recommend/discovery/{user_id}", response_model=List[int], tags=["Recommendations"])
async def get_user_discovery_recommendations(user_id: int = Path(..., title="用户ID", gt=0)):
    """
    获取用户的【猜你喜欢】推荐 (基于 ALS 隐向量)
    若无法计算 (新用户)，返回全局热门
    """
    key = f"{settings.REDIS_KEY_PREFIX}user:{user_id}:discovery"
    recs = await get_list_from_redis(key)
    if not recs:
         # 冷启动：无法预测则返回全局热门
        recs = await get_list_from_redis(f"{settings.REDIS_KEY_PREFIX}global:popular")
    return recs

@app.get("/recommend/related/{item_id}", response_model=List[int], tags=["Recommendations"])
async def get_item_related_recommendations(item_id: int = Path(..., title="工具ID", gt=0)):
    """
    获取工具的【相关推荐】 (Item-to-Item 相似度)
    """
    key = f"{settings.REDIS_KEY_PREFIX}item:{item_id}:related"
    return await get_list_from_redis(key)

@app.get("/health", tags=["System"])
async def health_check():
    """系统健康检查"""
    if not redis_client:
        return {"status": "error", "detail": "Redis uninitialized"}
    try:
        await redis_client.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    # 仅用于开发调试，生产环境建议直接使用 uvicorn 命令启动
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
