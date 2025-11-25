import implicit
import pandas as pd
import scipy.sparse as sparse
from typing import List, Dict
from config.settings import settings, logger

class RecommendationEngine:
    def __init__(self, 
                 user_map: Dict[int, int], 
                 item_inv_map: Dict[int, int],
                 history_df: pd.DataFrame):
        """
        :param user_map: DB_ID -> Matrix_Index
        :param item_inv_map: Matrix_Index -> DB_ID
        :param history_df: 预处理后的聚合数据，包含真实的分数 (for Scenario A)
        """
        self.user_map = user_map
        self.item_inv_map = item_inv_map
        self.history_df = history_df
        
        # 优化 A: 预构建用户历史索引 (O(1) 查询)
        # 按分数降序排序，然后聚合为列表
        logger.info("正在构建用户历史行为索引...")
        sorted_df = self.history_df.sort_values(['user_id', 'score'], ascending=[True, False])
        self.user_history_index = sorted_df.groupby('user_id')['object_id'].apply(list).to_dict()
        logger.info(f"用户索引构建完成，覆盖 {len(self.user_history_index)} 位用户")

        # 构建全局热门物品列表 (按总分降序)
        logger.info("正在计算全局热门物品...")
        popular_series = self.history_df.groupby('object_id')['score'].sum().sort_values(ascending=False)
        self.popular_items = popular_series.index.tolist()
        logger.info(f"热门物品计算完成，Top 5: {self.popular_items[:5]}")

        # 初始化 ALS 模型
        self.model = implicit.als.AlternatingLeastSquares(
            factors=settings.ALS_FACTORS,
            regularization=settings.ALS_REGULARIZATION,
            iterations=settings.ALS_ITERATIONS,
            alpha=settings.ALS_ALPHA,
            random_state=42  # 固定种子以便复现
        )
        self.is_trained = False
        self.user_items_matrix = None # 保存训练用的矩阵引用

    def train(self, user_item_matrix: sparse.csr_matrix):
        """
        训练 ALS 模型
        注意: implicit 0.7+ fit() 接受 user_item 矩阵 (N_users x N_items)
        """
        logger.info("开始训练 ALS 模型...")
        self.user_items_matrix = user_item_matrix
        # 训练
        self.model.fit(user_item_matrix)
        self.is_trained = True
        logger.info("模型训练完成")

    def get_history_rec(self, user_id: int) -> List[int]:
        """
        场景 A: 常用工具 (Frequency / History)
        不经过 ALS，直接取时间衰减加权后的 Top N
        """
        try:
            # 优化后：直接查字典，O(1) 复杂度
            all_items = self.user_history_index.get(user_id, [])
            return all_items[:settings.REC_HISTORY_COUNT]
        except Exception as e:
            logger.error(f"获取常用工具失败 User {user_id}: {e}")
            return []

    def get_discovery_rec(self, user_id: int) -> List[int]:
        """
        场景 B: 猜你喜欢 (Discovery / User-based)
        使用 ALS 向量积，且必须过滤掉用户历史行为
        """
        if not self.is_trained or user_id not in self.user_map:
            return []

        try:
            user_idx = self.user_map[user_id]
            
            # recommend 方法返回 (item_idx, score)
            # filter_already_liked_items=True 是关键，保证这是"发现"
            ids, scores = self.model.recommend(
                userid=user_idx, 
                user_items=self.user_items_matrix[user_idx], 
                N=settings.REC_DISCOVERY_COUNT, 
                filter_already_liked_items=True
            )
            
            # 将矩阵索引转回 DB ID
            # 注意: ids 可能是 numpy array
            rec_ids = [self.item_inv_map[idx] for idx in ids]
            return rec_ids
        except Exception as e:
            logger.error(f"获取猜你喜欢失败 User {user_id}: {e}")
            return []

    def get_popular_items(self, limit: int = 10) -> List[int]:
        """
        获取全局热门物品 (基于总分)
        """
        return self.popular_items[:limit]

    def get_related_items(self, item_id: int) -> List[int]:
        """
        场景 C: 相关推荐 (Item-to-Item)
        计算与给定物品最相似的其他物品
        """
        if not self.is_trained:
            return []
            
        # item_inv_map 是 idx -> id，我们需要 id -> idx
        # 由于 DataLoader 中有 item_id_to_idx，但在 Engine 初始化时未传入
        # 简单起见，这里通过 value 反查 key (在生产中应在 init 传入 item_map)
        # 或者我们可以遍历 item_inv_map，但太慢。
        # *修正*: 既然这是个优化任务，我们假设调用者会负责 ID 转换，或者我们在 Engine 里不做 ID 转换，
        # 但为了保持接口一致性 (输入输出都是 DB ID)，我们需要 item_map。
        # 由于 self.item_inv_map 是 idx->id，我们在这里临时构建一个 id->idx (为了不改动 __init__ 签名太大)
        if not hasattr(self, '_item_id_map_cache'):
            self._item_id_map_cache = {v: k for k, v in self.item_inv_map.items()}
        
        if item_id not in self._item_id_map_cache:
            return []
            
        item_idx = self._item_id_map_cache[item_id]
        
        try:
            # similar_items 返回 (item_idx, score)
            ids, scores = self.model.similar_items(itemid=item_idx, N=settings.REC_RELATED_COUNT + 1)
            
            # 过滤掉自己 (通常第一个是自己)
            rec_ids = []
            for idx in ids:
                if idx != item_idx:
                    rec_ids.append(self.item_inv_map[idx])
            
            return rec_ids[:settings.REC_RELATED_COUNT]
        except Exception as e:
            logger.error(f"获取相关推荐失败 Item {item_id}: {e}")
            return []