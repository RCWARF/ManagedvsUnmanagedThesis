"""Long-lived resources shared by all runs. Never destroyed between runs."""
from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sns as sns
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
        # Stack deletes run in the background (Lambdas in a VPC take a while to delete), so
        # CloudFormation's own EventBridge events say when a Rag* stack delete finished or failed.
        # Subscribe with `rag notify EMAIL`; the subscription lives outside CloudFormation so
        # redeploying this stack never drops it.
        self.notify = sns.Topic(self, "StackNotify", display_name="rag-thesis stacks")
        rule = events.Rule(
            self, "StackDeleteRule",
            description="Rag* stack delete finished or failed",
            event_pattern=events.EventPattern(
                source=["aws.cloudformation"],
                detail_type=["CloudFormation Stack Status Change"],
                detail={
                    "stack-id": [{"prefix": f"arn:aws:cloudformation:{self.region}:{self.account}:stack/Rag"}],
                    "status-details": {"status": ["DELETE_COMPLETE", "DELETE_FAILED"]},
                },
            ),
        )
        stack_id = events.EventField.from_path("$.detail.stack-id")
        status = events.EventField.from_path("$.detail.status-details.status")
        reason = events.EventField.from_path("$.detail.status-details.status-reason")
        rule.add_target(targets.SnsTopic(self.notify, message=events.RuleTargetInput.from_text(
            f"{status}: {stack_id} {reason}")))

        CfnOutput(self, "NotifyTopicArn", value=self.notify.topic_arn)
        CfnOutput(self, "DatasetsBucket", value=self.datasets.bucket_name, export_name="RagDatasetsBucket")
        CfnOutput(self, "ResultsBucket", value=self.results.bucket_name, export_name="RagResultsBucket")
