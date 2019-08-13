def join(*args):

    if len(args) == 1:  # must be the bucket
        return "/"+args+"/"

    return "/"+"/".join(args)
