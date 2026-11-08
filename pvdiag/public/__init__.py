"""公开数据：下载 / 转换 / 基准评测。

用公开的真实光伏数据给系统做"考试"，而不是只在合成电站上自说自话：
- kaggle_solar：Kaggle《Solar Power Generation Data》（印度 2 座电站 × 22 台逆变器，15 分钟 DC/AC 功率 + 辐照/组件温度，34 天）
- nrel_pvanalytics：NREL 真实逆变器片段（pvanalytics 自带，带人工标注：数据钉死 / 插值 / 离群 / 限发平顶）

`pvdiag public fetch` 下载，`pvdiag public load` 转成 pvdiag 的电站/逆变器/组串结构入库，`pvdiag bench` 跑全链路并打分。
"""
from .datasets import DATASETS, dataset_dir, fetch_dataset  # noqa: F401
