# pvdiag · 组串发电量异常诊断智能体系统
# 一条命令体验：docker compose up  → http://127.0.0.1:8080（自带合成电站 + 30 天回放）
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PVDIAG_CONFIG=/app/config.yaml
WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY pvdiag ./pvdiag
RUN pip install --upgrade pip && pip install -e ".[web,dev,bench]"

COPY tests ./tests
COPY config.example.yaml ./
COPY deploy ./deploy
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/app/data"]
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
