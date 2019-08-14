import os

# credentials
assert 'credentials' in os.listdir(os.path.join(os.environ['HOME'], '.aws')), "Set up your credentials in ~/.aws/credentials (see boto3 docs)"

from globals import training_bucket
import data

# training bucket is available
assert training_bucket in data.os.listdir(), "Training bucket not found. Set globals.training_bucket, example:  \"/my_bucket_name\""