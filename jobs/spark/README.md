# Spark job 开发指南

按任务职责组织代码，算法公式、任务编排和运行基础设施分开维护：

| 目录 | 职责 |
| --- | --- |
| `recall/` | 召回公式、任务注册、累计快照输入、输出发布和命令构建 |
| `ranking/` | 排序样本、逐标签 point-in-time 特征、训练编排和命令构建 |
| `analytics/` | 业务指标聚合和命令构建 |
| `common/` | Hive/HDFS 数据读写、Spark session、spark-submit 公共参数 |
| `runner.py` | HTTP 网关、任务路由、并发锁、超时和结果解析 |

`recall/` 的公式进一步按 popularity（热门/最新）、collaborative（协同过滤）、
content（内容相似/搜索文档）、embedding（Word2Vec/ALS）、title_embedding（标题向量）分类。
`catalog.py` 是召回算法名单、处理函数、发布表名和 runner 默认算法参数的唯一来源。
网关及 command/catalog 模块不导入 PySpark 或训练依赖。

## 新增召回算法

1. 在对应公式模块中实现 DataFrame → DataFrame 的计算函数；独立算法也可以新建模块。
2. 实现 `compute(inputs, args)` 适配函数。`inputs.items/users/events` 按需读取一次累计快照，
   `events` 已过滤被删除的 item。用户内容算法可用 `inputs.users_by_scene()`，按行为类型
   筛选场景时传入 `args.event_type`。不要在算法内重写 tombstone/快照逻辑。
3. 在 `recall/catalog.py` 的 `RECALL_JOBS` 添加一项，例如复用热门公式开发试验任务：

```python
# recall/custom.py
from .popularity import hot


def compute(inputs, args):
    return hot(inputs.events, args.size, args.event_type)


# recall/catalog.py 的 RECALL_JOBS 中增加：
# "custom_hot": RecallSpec(
#     "jobs.spark.recall.custom:compute", "hot", default_size=100,
# )
```

随后可直接调用既有 CLI 或 runner 的 `/jobs/recall`，无需增加算法分发分支：

```bash
spark-submit --master local[2] --py-files /path/to/rec-algorithm.zip \
  jobs/spark/recall/job.py custom_hot --date 2026-10-18 \
  --output-path file:///tmp/custom-hot
```

返回结果不要自行添加 `dt`、写文件或发布；`recall/job.py` 统一添加业务日期、写结果并
调用 `recall/output.py`。外部请求生成的向量等昂贵结果可设置 `persist=True`，公共流程
会在写入/发布成功或失败后释放缓存。`publisher="embedding"` 使用序列向量发布协议，
默认 `publisher="recall"` 使用召回发布协议。

新增参数在 `recall/job.py::parser()` 声明类型和 CLI 默认值；需要从 runner 传入时，在
对应 `RecallSpec.command_options` 增加 `(参数名, runner默认值)`，名称用 snake_case。
CLI 和调度默认值沿用历史差异，例如 CLI 的 `min_count=5`，runner 为 1。

输出须遵守现有 serving schema。示例复用 `hot` 发布表，只用于说明扩展方法；实际新增
发布类型还需配套 publisher、rec-console 校验/激活、服务端召回配置、Airflow/UI 和分发
契约检查。仅写 Parquet 的算法开发可以先不使用 `--publish`。

## 新增其他类型 job

在对应类别下提供 `parser()` 和 `run(args, spark=None)`；由 `common.session.get_spark`
创建或复用 session，读取实体使用 `common.io`。排序训练应复用 `ranking/point_in_time.py`
以保留严格历史时点语义，不能用最新快照代替逐标签特征。

需要调度时，在类别内的 `command.py` 构建参数数组，复用 `common.submission.spark_submit`，
让分类目录中的 `job.py` 可直接作为 CLI 脚本执行（使用绝对导入）。将路径和 command builder 加入 `runner.JOB_ROUTES`；若需要结构化
结果，声明 stdout 标记、响应字段及缺失结果错误。新增 HTTP 路径需要同步调用方。

传入的 Spark session 由调用者管理，job 不停止它；命令行进程结束时释放自己的 session。
不要在模块导入阶段创建 session、访问网络或读取线上数据。

## 兼容性与验证

任务入口统一为 `recall/job.py`、`ranking/job.py` 和 `analytics/job.py`。
runner 直接提交这些脚本，`openrec-spark-recall` 命令直接指向 `recall.job:main`。
`python -m jobs.spark.runner`、HTTP 接口、任务参数及输出协议保持一致。
旧顶层脚本和导入转发已删除；手工 Spark 提交需改用新脚本路径，测试直接引用实际模块。

```bash
pytest -q test/jobs/spark/test_runner.py test/jobs/spark/test_catalog.py \
  test/jobs/spark/test_title_embedding.py
# 以下需要 PySpark 和兼容 JDK；不需要在线 Redis/ES/Kafka。
pytest -q test/jobs/spark/test_job_pipeline.py test/jobs/spark/test_recall_parity.py \
  test/jobs/spark/test_point_in_time_values.py
spark-submit --master local[2] test/jobs/spark/smoke_text_search.py
```

新增算法应覆盖公式及输出 schema；新增编排应覆盖输入选择、参数传递和失败清理。
`smoke_title_embedding.py` 另需真实 vLLM、Elasticsearch 和 rec-console，是显式集成测试。
