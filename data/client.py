# Resource to use

import boto3
from globals import region

s3 = boto3.client('s3', region_name=region)
