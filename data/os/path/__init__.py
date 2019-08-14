def join(*args):

    if args == ():
        return "/"

    is_dict = True if args[-1][-1] == "/" else False

    path = "/"+"/".join([a.strip("/") for a in args])

    if is_dict:
        path = path + "/"

    return path
