"""Long-lived resources shared by all runs. Never destroyed between runs."""
from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_s3 as s3
from constructs import Construct


class RagCommonStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)
        common = dict(
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        self.datasets = s3.Bucket(self, "Datasets", versioned=True, **common)
        self.results = s3.Bucket(
            self, "Results", versioned=True,
            lifecycle_rules=[s3.LifecycleRule(noncurrent_version_expiration=Duration.days(90))],
            **common,
        )
        CfnOutput(self, "DatasetsBucket", value=self.datasets.bucket_name, export_name="RagDatasetsBucket")
        CfnOutput(self, "ResultsBucket", value=self.results.bucket_name, export_name="RagResultsBucket")
