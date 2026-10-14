"""Variant A: S3 -> Enqueue Lambda -> SQS FIFO -> Embed Lambda -> Titan V2 -> S3 Vectors;
query Lambda -> Titan V2 -> S3 Vectors -> Nova Lite (in-region).

One stack per *infrastructure* configuration (RunConfig.infra_spec: Lambda specs, network,
model ids), deployed once and reused by every run of a sweep. Attaching Lambdas to the VPC
makes create and delete slow, so this stack is not rebuilt per run. Per run, the harness
creates a fresh vector index in this stack's vector bucket and applies the run's settings
(environment, reserved concurrency, ESM scaling) from the RunConfig through the API.
"""
from pathlib import Path

from aws_cdk import CfnOutput, Duration, Fn, RemovalPolicy, Stack, Tags
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as les
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_s3vectors as s3v
from aws_cdk import aws_sqs as sqs
from constructs import Construct

from rag_config import RunConfig

LAMBDA_SRC = Path(__file__).resolve().parents[2] / "lambdas" / "src"
RUNTIMES = {"python3.13": lambda_.Runtime.PYTHON_3_13, "python3.14": lambda_.Runtime.PYTHON_3_14}
ARCHES = {"arm64": lambda_.Architecture.ARM_64, "x86_64": lambda_.Architecture.X86_64}


class VariantAStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: RunConfig,
                 vpc: ec2.IVpc | None = None, lambda_sg: ec2.ISecurityGroup | None = None, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)
        infra_key = cfg.infra_hash()
        region = self.region

        for k, v in {"Project": "rag-thesis", "Variant": cfg.variant_id, "InfraKey": infra_key}.items():
            Tags.of(self).add(k, v)

        # ---- vector store: indexes are created per run by the harness ----------
        vbucket = s3v.CfnVectorBucket(self, "VectorBucket")
        index_arns = f"{vbucket.attr_vector_bucket_arn}/index/*"

        # ---- Bedrock models (in-region; optional application profiles for cost tags) ----
        def model_ref(name: str, model_id: str, use_profile: bool) -> tuple[str, list[str]]:
            fm_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
            if not use_profile:
                return model_id, [fm_arn]
            profile = bedrock.CfnApplicationInferenceProfile(
                self, f"{name}Profile",
                inference_profile_name=f"rag-a-{infra_key}-{name.lower()}",
                model_source=bedrock.CfnApplicationInferenceProfile.InferenceProfileModelSourceProperty(copy_from=fm_arn),
                tags=[{"key": "InfraKey", "value": infra_key}, {"key": "Project", "value": "rag-thesis"}],
            )
            return profile.attr_inference_profile_arn, [fm_arn, profile.attr_inference_profile_arn]

        embed_model, embed_arns = model_ref("Embed", cfg.embedding.model_id, cfg.embedding.use_app_profile)
        gen_model, gen_arns = model_ref("Gen", cfg.generation.model_id, cfg.generation.use_app_profile)

        # ---- queue + ingest bucket ----------------------------------------
        dlq = sqs.Queue(self, "Dlq", fifo=True, retention_period=Duration.days(4))
        queue = sqs.Queue(
            self, "IngestQueue",
            fifo=True,
            # High-throughput FIFO: limits apply per message group (one group per doc).
            fifo_throughput_limit=sqs.FifoThroughputLimit.PER_MESSAGE_GROUP_ID,
            deduplication_scope=sqs.DeduplicationScope.MESSAGE_GROUP,
            # AWS guidance: at least 6x the consumer function timeout.
            visibility_timeout=Duration.seconds(6 * cfg.lambdas.embed.timeout_s),
            dead_letter_queue=sqs.DeadLetterQueue(queue=dlq, max_receive_count=3),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
        )
        ingest_bucket = s3.Bucket(
            self, "IngestBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # ---- Lambdas --------------------------------------------------------
        # Static environment only; the harness merges RunConfig.run_env() over it per run.
        env = {"VECTOR_STORE": "s3v", "EMBED_MODEL": embed_model, "GEN_MODEL": gen_model,
               "QUEUE_URL": queue.queue_url, "INFRA_KEY": infra_key}
        code = lambda_.Code.from_asset(str(LAMBDA_SRC), exclude=["**/__pycache__", "**/*.pyc"])
        net = {}
        if cfg.network.in_vpc:
            if vpc is None or lambda_sg is None:
                raise ValueError("network.in_vpc is true but no RagNetwork stack was passed")
            net = dict(vpc=vpc, security_groups=[lambda_sg],
                       vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED))

        def fn(name: str, handler: str, spec) -> lambda_.Function:
            # Default log group (/aws/lambda/<function>), shared by the runs that use this stack;
            # every SAAF record carries run_id and collect filters on it.
            return lambda_.Function(
                self, name,
                runtime=RUNTIMES[spec.runtime],
                architecture=ARCHES[spec.arch],
                handler=handler,
                code=code,
                memory_size=spec.memory_mb,
                timeout=Duration.seconds(spec.timeout_s),
                environment=env,
                logging_format=lambda_.LoggingFormat.TEXT,
                **net,
            )

        enqueue = fn("Enqueue", "enqueue_handler.lambda_handler", cfg.lambdas.enqueue)
        # Reserved concurrency, ESM batch size and ESM scaling are the swept variables, so the
        # harness sets them per run from the RunConfig (deploy.configure), not CloudFormation.
        embed = fn("Embed", "embed_handler.lambda_handler", cfg.lambdas.embed)
        query = fn("Query", "query_handler.lambda_handler", cfg.lambdas.query)

        queue.grant_send_messages(enqueue)
        ingest_bucket.grant_read(embed)
        ingest_bucket.add_event_notification(s3.EventType.OBJECT_CREATED, s3n.LambdaDestination(enqueue))
        esm = les.SqsEventSource(queue, batch_size=1, report_batch_item_failures=True)
        embed.add_event_source(esm)

        s3v_write = iam.PolicyStatement(actions=["s3vectors:PutVectors", "s3vectors:GetVectors"],
                                        resources=[index_arns])
        s3v_read = iam.PolicyStatement(actions=["s3vectors:QueryVectors", "s3vectors:GetVectors"],
                                       resources=[index_arns])
        invoke_embed = iam.PolicyStatement(actions=["bedrock:InvokeModel"], resources=embed_arns)
        invoke_gen = iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], resources=gen_arns)
        for stmt in (s3v_write, invoke_embed):
            embed.add_to_role_policy(stmt)
        for stmt in (s3v_read, invoke_embed, invoke_gen):
            query.add_to_role_policy(stmt)

        if cfg.query.entrypoint == "function_url":
            url = query.add_function_url(auth_type=lambda_.FunctionUrlAuthType.AWS_IAM,
                                         invoke_mode=lambda_.InvokeMode.BUFFERED)
            CfnOutput(self, "OutQueryUrl", value=url.url)

        for name, value in {
            "InfraKey": infra_key, "IngestBucket": ingest_bucket.bucket_name,
            "QueueUrl": queue.queue_url, "DlqUrl": dlq.queue_url,
            "VectorBucket": vbucket.attr_vector_bucket_arn,
            "VectorBucketName": Fn.select(1, Fn.split("/", vbucket.attr_vector_bucket_arn)),
            "EnqueueFn": enqueue.function_name, "EmbedFn": embed.function_name,
            "QueryFn": query.function_name, "EmbedEsm": esm.event_source_mapping_id,
            "EmbedModel": embed_model, "GenModel": gen_model,
        }.items():
            CfnOutput(self, f"Out{name}", value=value)
