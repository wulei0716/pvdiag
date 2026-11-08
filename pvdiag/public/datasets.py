"""公开数据集登记与下载（带 sha256 校验、断点缓存）。

只依赖 requests；下载源选择 GitHub 原始文件镜像（CI 与多数内网代理都放行 github），
每个文件登记 sha256，下载后校验，防止镜像被改动或半成品文件。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PublicFile:
    name: str
    url: str
    sha256: str
    size: int


@dataclass
class PublicDataset:
    key: str
    title: str
    license_note: str
    files: List[PublicFile] = field(default_factory=list)
    source: str = ""


KAGGLE_MIRROR = "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Solar%20Plant/"

DATASETS: Dict[str, PublicDataset] = {
    "kaggle_solar": PublicDataset(
        key="kaggle_solar",
        title="Kaggle · Solar Power Generation Data（印度 2 座电站 × 22 台逆变器，2020-05-15 ~ 2020-06-17，15 分钟）",
        license_note="Kaggle 公开数据集（anikannal/solar-power-generation-data），公开镜像；仅用于算法验证",
        source="https://www.kaggle.com/datasets/anikannal/solar-power-generation-data",
        files=[
            PublicFile("Plant_1_Generation_Data.csv", KAGGLE_MIRROR + "Plant_1_Generation_Data.csv",
                       "7573be47df0bcd19d203e8ed1504faf0ab805ecb9a8fd44bb5de0d886c698750", 4770297),
            PublicFile("Plant_1_Weather_Sensor_Data.csv", KAGGLE_MIRROR + "Plant_1_Weather_Sensor_Data.csv",
                       "91325041328745c288cb809aebd8b1b59cde54284500902ead1fb393499d6806", 287847),
            PublicFile("Plant_2_Generation_Data.csv", KAGGLE_MIRROR + "Plant_2_Generation_Data.csv",
                       "9b735e9278774d4d1dd4601047315ef50b3793244d892446f6aabf4fae63bd37", 5805157),
            PublicFile("Plant_2_Weather_Sensor_Data.csv", KAGGLE_MIRROR + "Plant_2_Weather_Sensor_Data.csv",
                       "cafd80885a51b88521b3301cffa4e4f013e2516fcb3478b439a81855edac3a64", 301443),
        ],
    ),
    # NREL 片段随 pvanalytics 包分发（pip install pvanalytics），无需单独下载
    "nrel_pvanalytics": PublicDataset(
        key="nrel_pvanalytics",
        title="NREL 真实逆变器片段（pvanalytics 自带：ac_power_inv_2173 钉死标注、ac_power_inv_7539 限发标注）",
        license_note="pvanalytics (MIT) 随包分发的 NREL 数据；pip install pvanalytics 即得",
        source="https://github.com/pvlib/pvanalytics/tree/main/pvanalytics/data",
    ),
}


def dataset_dir(key: str, root: Optional[os.PathLike | str] = None) -> Path:
    """缓存目录：$PVDIAG_PUBLIC_DIR 或 ./data/public/<key>"""
    base = Path(root) if root else Path(os.environ.get("PVDIAG_PUBLIC_DIR", "./data/public"))
    return base / key


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_cached(key: str, root: Optional[os.PathLike | str] = None) -> bool:
    ds = DATASETS[key]
    d = dataset_dir(key, root)
    return all((d / f.name).exists() and (d / f.name).stat().st_size == f.size for f in ds.files)


def fetch_dataset(key: str, root: Optional[os.PathLike | str] = None, force: bool = False, timeout: int = 180) -> Path:
    """下载并校验；已存在且校验通过则直接返回。失败抛异常（调用方决定是否降级）。"""
    import requests
    ds = DATASETS[key]
    d = dataset_dir(key, root)
    d.mkdir(parents=True, exist_ok=True)
    for f in ds.files:
        p = d / f.name
        if p.exists() and not force and p.stat().st_size == f.size and sha256_of(p) == f.sha256:
            continue
        tmp = p.with_suffix(p.suffix + ".part")
        with requests.get(f.url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp, "wb") as out:
                for chunk in r.iter_content(1 << 16):
                    out.write(chunk)
        got = sha256_of(tmp)
        if got != f.sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{f.name} sha256 不符：期望 {f.sha256[:12]}… 实得 {got[:12]}…（镜像文件已变？）")
        tmp.replace(p)
    return d
