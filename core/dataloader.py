import pandas as pd
import scipy.sparse as sparse
from sqlalchemy import create_engine, text
from typing import Tuple, Dict
from config.settings import settings, logger

class DataLoader:
    """
    负责从数据库加载原始数据，处理时间衰减，并生成稀疏矩阵。
    """

    def __init__(self):
        self.engine = create_engine(settings.DB_URL)
        # 映射字典
        self.user_id_to_idx: Dict[int, int] = {}
        self.user_idx_to_id: Dict[int, int] = {}
        self.item_id_to_idx: Dict[int, int] = {}
        self.item_idx_to_id: Dict[int, int] = {}

    def load_and_process(self) -> Tuple[pd.DataFrame, sparse.csr_matrix]:
        """
        核心流程：
        1. 读取 DB
        2. 时间衰减计算 Score
        3. 生成 CSR Matrix
        
        Returns:
            df_grouped: 包含 user_id, object_id, score 的聚合 DataFrame (用于场景A)
            sparse_user_item: 用户-物品 稀疏矩阵 (用于场景B ALS输入)
        """
        logger.info("开始加载数据库数据...")
        
        # 1. SQL 查询：只读取必要字段，过滤已删除和超时的记录
        # 注意：虽然可以在 SQL 做衰减，但 Pandas 处理复杂逻辑更灵活，且 56w 数据量完全可控
        query = text(f"""
            SELECT user_id, object_id, visited_at
            FROM pre_browser_histories
            WHERE deleted_at IS NULL
              AND visited_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
        """)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"days": settings.TIME_DECAY_WINDOW})
        except Exception as e:
            logger.error(f"数据库读取失败: {e}")
            raise

        if df.empty:
            logger.warning("未查询到有效数据！")
            return pd.DataFrame(), sparse.csr_matrix((0, 0))

        logger.info(f"原始数据行数: {len(df)}")

        # 2. Pandas 矢量化计算时间衰减
        # Score = sum( decay_rate ^ days_diff )
        now = pd.Timestamp.now()
        # 计算距离今天的天数 (float)
        df['days_diff'] = (now - pd.to_datetime(df['visited_at'])).dt.total_seconds() / (24 * 3600)
        # 应用衰减公式
        df['weight'] = settings.TIME_DECAY_RATE ** df['days_diff']
        
        # 聚合：同一用户对同一工具的多次访问权重求和
        df_grouped = df.groupby(['user_id', 'object_id'])['weight'].sum().reset_index()
        df_grouped.rename(columns={'weight': 'score'}, inplace=True)
        
        logger.info(f"聚合后交互对数量: {len(df_grouped)}")

        # 3. 建立映射 (ID Mapping)
        unique_users = df_grouped['user_id'].unique()
        unique_items = df_grouped['object_id'].unique()

        self.user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.user_idx_to_id = {i: uid for i, uid in enumerate(unique_users)}
        self.item_id_to_idx = {iid: i for i, iid in enumerate(unique_items)}
        self.item_idx_to_id = {i: iid for i, iid in enumerate(unique_items)}

        # 4. 构建稀疏矩阵 (CSR Format)
        # implicit 推荐时通常需要 User-Item 形式
        users_indices = df_grouped['user_id'].map(self.user_id_to_idx).values
        items_indices = df_grouped['object_id'].map(self.item_id_to_idx).values
        scores = df_grouped['score'].values

        # 形状: (N_Users, N_Items)
        sparse_user_item = sparse.csr_matrix(
            (scores, (users_indices, items_indices)), 
            shape=(len(unique_users), len(unique_items))
        )

        logger.info(f"稀疏矩阵构建完成. Shape: {sparse_user_item.shape}")
        
        return df_grouped, sparse_user_item