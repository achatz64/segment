import os
from segment.globals import training_bucket
import segment.data as data

# credentials
def credentials():
    assert 'credentials' in os.listdir(os.path.join(os.environ['HOME'], '.aws')), "Set up your credentials in ~/.aws/credentials (see boto3 docs)"

# training bucket is available
def training_bucket():
    assert training_bucket in data.os.listdir(), "Training bucket not found. Set globals.training_bucket, example:  \"/my_bucket_name\""