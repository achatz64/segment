from data.client import s3, region

from data.os import path

def listdir(abspath="/"):

    assert abspath[0] == "/", "Use absolute paths"

    if abspath == "/":
        return [bucket["Name"] for bucket in s3.list_buckets().get('Buckets', [])]

    split = abspath.split("/")

    bucket = split[1]
    rest = "/".join(split[2:])

    bucket_contents = [content["Key"] for content in s3.list_objects(Bucket=bucket).get('Contents', [])]
    filter_contents = [content for content in bucket_contents if content[:len(rest)] == rest]

    return filter_contents


def mkbucket(name, region=region, ACL='private'):

    exist = listdir('')

    if name in exist:
        raise FileExistsError

    location = {'LocationConstraint': region}

    s3.create_bucket(Bucket=name, CreateBucketConfiguration=location, ACL=ACL)

