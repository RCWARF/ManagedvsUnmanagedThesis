"""boto3 clients at module scope, so warm invocations reuse connections.

botocore's own retries are switched off (max_attempts=1): every throttle reaches
our code and is counted, instead of being silently retried underneath it.
"""
import boto3
from botocore.config import Config

from .settings import SETTINGS

_NO_SDK_RETRIES = Config(
    retries={"max_attempts": 1, "mode": "standard"},
    connect_timeout=5,
    read_timeout=60,
    max_pool_connections=max(10, SETTINGS.embed_workers * 2),
)

s3 = boto3.client("s3", region_name=SETTINGS.region, config=_NO_SDK_RETRIES)
sqs = boto3.client("sqs", region_name=SETTINGS.region, config=_NO_SDK_RETRIES)
bedrock = boto3.client("bedrock-runtime", region_name=SETTINGS.region, config=_NO_SDK_RETRIES)
s3vectors = boto3.client("s3vectors", region_name=SETTINGS.region, config=_NO_SDK_RETRIES)
