import sys
import time
import redis
from typing import List, Dict
from core.data_loader import DataLoader
from core.engine import RecommendationEngine
from config.settings import logger, settings

def save_results_to_redis(user_recs: Dict[int, Dict[str, List[int]]], item_recs: Dict[int, List[int]]):
    """
    将结果写入 Redis。使用 Pipeline 提高吞吐量。
    User Recs Keys:
      - rec:sys:user:{uid}:history -> List[ItemID]
      - rec:sys:user:{uid}:discovery -> List[ItemID]
    Item Recs Keys:
      - rec:sys:item:{iid}:related -> List[ItemID]
    """
    logger.info(f"正在保存推荐结果到 Redis ({settings.REDIS_URL})...")
    
    try:
        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        pipe = r.pipeline()
        
        # 1. 保存用户推荐
        logger.info(f"保存 {len(user_recs)} 位用户的推荐列表")
        for uid, recs in user_recs.items():
            # History
            key_hist = f"{settings.REDIS_KEY_PREFIX}user:{uid}:history"
            pipe.delete(key_hist)
            if recs['history']:
                pipe.rpush(key_hist, *recs['history'])
                pipe.expire(key_hist, settings.REDIS_EXPIRE_SECONDS)
            
            # Discovery
            key_disc = f"{settings.REDIS_KEY_PREFIX}user:{uid}:discovery"
            pipe.delete(key_disc)
            if recs['discovery']:
                pipe.rpush(key_disc, *recs['discovery'])
                pipe.expire(key_disc, settings.REDIS_EXPIRE_SECONDS)
        
        # 2. 保存物品相关推荐
        logger.info(f"保存 {len(item_recs)} 个物品的相关推荐")
        for iid, related_ids in item_recs.items():
            key_related = f"{settings.REDIS_KEY_PREFIX}item:{iid}:related"
            pipe.delete(key_related)
            if related_ids:
                pipe.rpush(key_related, *related_ids)
                pipe.expire(key_related, settings.REDIS_EXPIRE_SECONDS)
        
        # 执行 Pipeline
        # 实际生产中如果数据量过大(>10w)，应分批 execute，这里为演示一次性提交
        pipe.execute()
        logger.info("Redis 写入完成。")
        
    except Exception as e:
        logger.error(f"Redis 保存失败: {e}")
        # 不阻断主流程，但记录错误

def main():
    start_time = time.time()
    logger.info("=== 推荐系统离线任务启动 ===")

    # 1. 数据准备
    loader = DataLoader()
    try:
        df_grouped, sparse_matrix = loader.load_and_process()
        if sparse_matrix.shape[0] == 0:
            logger.error("没有数据，任务终止")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"数据加载阶段严重错误: {e}")
        sys.exit(1)

    # 2. 模型训练
    engine = RecommendationEngine(
        user_map=loader.user_id_to_idx,
        item_inv_map=loader.item_idx_to_id,
        history_df=df_grouped
    )
    
    try:
        engine.train(sparse_matrix)
    except Exception as e:
        logger.critical(f"模型训练失败: {e}")
        sys.exit(1)

    # 3. 批量预测 (为所有活跃用户生成推荐)
    active_users = df_grouped['user_id'].unique()
    logger.info(f"开始为 {len(active_users)} 位活跃用户计算推荐结果...")

    user_recommendations = {}
    for uid in active_users:
        rec_history = engine.get_history_rec(uid)
        rec_discovery = engine.get_discovery_rec(uid)
        user_recommendations[uid] = {
            "history": rec_history,
            "discovery": rec_discovery
        }

    # 4. 物品相关推荐计算 (Item-to-Item)
    active_items = df_grouped['object_id'].unique()
    logger.info(f"开始为 {len(active_items)} 个工具计算相关推荐...")
    
    item_recommendations = {}
    for iid in active_items:
        rec_related = engine.get_related_items(iid)
        item_recommendations[iid] = rec_related

    # 5. 结果落库
    save_results_to_redis(user_recommendations, item_recommendations)

    elapsed = time.time() - start_time
    logger.info(f"=== 任务完成，耗时: {elapsed:.2f} 秒 ===")

if __name__ == "__main__":
    main()