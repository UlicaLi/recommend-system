# Tool Recommendation System (RecSys)

一个用于工具类网站的生产级推荐系统离线计算引擎。基于用户的历史行为数据，利用协同过滤算法生成个性化推荐，并将结果存储至 Redis 供在线业务实时调用。

## 🚀 核心功能

系统通过离线批处理任务（Batch Job）提供以下三种维度的推荐能力：

1.  **常用工具 (Recent History)**
    *   **场景**: 首页 "最近使用" / "常访问"。
    *   **逻辑**: 基于用户历史访问记录，采用 **时间衰减 (Time Decay)** 算法计算权重。距离现在越近的访问权重越高 ($Score = \sum 0.95^{\Delta days}$)，精准捕捉用户短期兴趣。
    *   **特点**: 内存字典索引加速，查询性能 O(1)。

2.  **猜你喜欢 (Discovery)**
    *   **场景**: 首页 "为您推荐" / 侧边栏。
    *   **逻辑**: 使用 **ALS (Alternating Least Squares)** 隐因子模型。通过矩阵分解挖掘用户和工具的潜在特征向量，预测用户可能感兴趣但从未交互过的工具。
    *   **特点**: 自动过滤用户已使用过的工具，保证推荐的新颖性。

3.  **相关推荐 (Item-to-Item)**
    *   **场景**: 工具详情页 "看了这个的人也看了..."。
    *   **逻辑**: 基于 ALS 模型生成的物品隐向量，计算 **余弦相似度 (Cosine Similarity)**，找出语义上最相似的工具集合。
    
4.  **热门推荐 (Popular Items)**
    *   **场景**: 冷启动 (新用户、无历史记录的用户) / 全局热门展示。
    *   **逻辑**: 基于所有用户历史行为数据，按工具的总交互分数进行降序排序。
    *   **特点**: 预计算并存储，用于提供默认或广泛受欢迎的推荐。

## 🛠 技术架构

*   **语言**: Python 3.10+
*   **算法核心**: [implicit](https://github.com/benfred/implicit) (高性能 ALS 实现)
*   **数据工程**: Pandas (矢量化 ETL), Scipy (Sparse Matrix)
*   **数据存储**:
    *   **Source**: MySQL (通过 SQLAlchemy 读取)
    *   **Sink**: Redis (Pipeline 批量写入)

## 📂 目录结构

```text
.
├── config/
│   └── settings.py    # 统一配置中心 (数据库、Redis、算法超参数、业务规则)
├── core/
│   ├── dataloader.py  # 数据加载器：执行 SQL、时间衰减计算、构建 CSR 稀疏矩阵
│   └── engine.py      # 推荐引擎：封装 ALS 模型训练、User/Item 向量检索与推理
├── main.py            # 程序入口：编排 ETL -> Train -> Predict -> Store 全流程
├── pyproject.toml     # 项目依赖定义
└── logs/              # 运行日志
```

## ⚙️ 配置与运行

### 1. 安装依赖

本项目使用 `uv` 或标准的 `pip` 进行包管理。

```bash
# 使用 pip
pip install .

# 使用uv
uv venv
uv sync
```

### 2. 环境变量配置

系统优先读取环境变量。你可以在运行前 export 变量，或者修改 `config/settings.py` 的默认值。

| 变量名 | 描述 | 示例 |
| :--- | :--- | :--- |
| `DB_URL` | MySQL 连接字符串 | `mysql+pymysql://user:pass@localhost:3306/db` |
| `REDIS_URL` | Redis 连接字符串 | `redis://localhost:6379/0` |

### 3. 运行离线任务

```bash
python main.py
```

### 4. 存储结果格式 (Redis)

任务运行成功后，数据将以 `List` 结构存入 Redis，TTL 默认为 24 小时：

*   **用户-历史**: `rec:sys:user:{user_id}:history` (value: `[item_id_1, item_id_2, ...]`)
*   **用户-发现**: `rec:sys:user:{user_id}:discovery`
*   **物品-相关**: `rec:sys:item:{item_id}:related`
*   **全局-热门**: `rec:sys:global:popular`

## 📈 性能优化

*   **查询优化**: 使用 Hash Map 替代 DataFrame 过滤，实现 O(1) 的用户历史查询。
*   **批量写入**: 使用 Redis Pipeline 技术，大幅降低网络 RTT，提升大规模数据落库速度。
*   **数据类型兼容**: 在写入 Redis 前，将 Pandas/NumPy 产生的 `int64` 等数据类型统一转换为 Python 原生 `int`，确保 Redis 存储的稳定性与兼容性，避免数据类型错误。
