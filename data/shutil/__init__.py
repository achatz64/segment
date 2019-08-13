from data.client import s3
import os as oldos

def copyfile(src, dst):

    if not oldos.path.exists(src):
        raise FileNotFoundError

    assert dst[0] == "/", "Use absolute paths"
    assert dst != "/", "Specify bucket"

    split = dst.split("/")

    bucket = split[1]
    rest = "/".join(split[2:])

    if rest == "":
        object_name = rest + oldos.path.split(src)[1]
    elif rest[-1]!="/":
        object_name = rest
    else:
        object_name = rest + oldos.path.split(src)[1]

    s3.upload_file(src, bucket, object_name)