import boto3
from botocore.config import Config

from .paths import REGION

_cfg = Config(region_name=REGION, retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=128)


def client(name: str):
    return boto3.client(name, config=_cfg)


def stack_outputs(stack_name: str) -> dict:
    desc = client("cloudformation").describe_stacks(StackName=stack_name)["Stacks"][0]
    return {o["OutputKey"].removeprefix("Out"): o["OutputValue"] for o in desc.get("Outputs", [])}
