"""Per-session network: deployed when a batch of runs starts, destroyed when it ends.

Every variant's Lambdas run in this VPC so the network path to Bedrock, SQS and the
vector store is the same for all of them (the OpenSearch-on-EC2 variant needs a VPC
anyway). No NAT gateway: all AWS traffic goes through endpoints in-region.
Cost while deployed: 3 interface endpoints x 2 AZs x $0.01/h (us-west-2 price list).
"""
from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_ec2 as ec2
from constructs import Construct


class RagNetworkStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)
        self.vpc = ec2.Vpc(
            self, "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="private", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=22),
            ],
        )
        self.lambda_sg = ec2.SecurityGroup(self, "LambdaSg", vpc=self.vpc, allow_all_outbound=True,
                                           description="RAG Lambdas")
        endpoint_sg = ec2.SecurityGroup(self, "EndpointSg", vpc=self.vpc, allow_all_outbound=False)
        endpoint_sg.add_ingress_rule(self.lambda_sg, ec2.Port.tcp(443), "Lambdas to endpoints")

        self.vpc.add_gateway_endpoint("S3", service=ec2.GatewayVpcEndpointAwsService.S3)
        for name, svc in (
            ("BedrockRuntime", ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME),
            ("Sqs", ec2.InterfaceVpcEndpointAwsService.SQS),
            ("S3Vectors", ec2.InterfaceVpcEndpointAwsService.S3_VECTORS),
        ):
            self.vpc.add_interface_endpoint(name, service=svc, private_dns_enabled=True,
                                            security_groups=[endpoint_sg])

        CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
        # Keep the exports RagA-* stacks import even when no RagA stack is in the CDK app
        # (`rag network up` synthesizes RagNetwork alone). Without this, CDK would try to drop
        # exports that a live or deleting RagA stack still uses, and the update rolls back.
        self.export_value(self.lambda_sg.security_group_id)
        for subnet in self.vpc.isolated_subnets:
            self.export_value(subnet.subnet_id)
