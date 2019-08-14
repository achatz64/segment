from data.client import s3
import os as oldos


def postfile(src, dst):

    if not oldos.path.exists(src):
        raise FileNotFoundError

    assert dst[0] == "/", "Use absolute paths"
    assert dst != "/", "Specify bucket"

    split = dst.split("/")

    bucket = split[1]
    rest = "/".join(split[2:])

    if rest == "":
        object_name = rest + oldos.path.split(src)[1]
    elif rest[-1] != "/":
        object_name = rest
    else:
        object_name = rest + oldos.path.split(src)[1]

    s3.upload_file(src, bucket, object_name)


def getfile(src, dst):

    assert src[0] == "/", "Use absolute paths"
    assert src != "/", "Specify bucket"
    assert src[-1] != "/", "Source is a \"directory\""

    split = src.split("/")

    bucket = split[1]
    rest = "/".join(split[2:])

    if rest == "":
        raise FileNotFoundError

    if oldos.path.isdir(dst):
        dst = oldos.path.join(dst, split[-1])

    s3.download_file(bucket, rest, dst)
