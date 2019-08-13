# Resource to use

import boto3

region = "eu-central-1"
s3 = boto3.client('s3', region_name=region)
