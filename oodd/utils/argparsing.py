import argparse
import json

from json import JSONDecoder


def json_or_str(arg, json_decoder=json.loads):
    """Parse inputs that can be either json or string as json if it looks like JSON otherwise string"""
    is_json = len(arg.split("{")) > 1 or len(arg.split("[")) > 1
    if is_json:
        return json_decoder(arg)
    return arg.split(" ")


def json_file_or_json(arg, json_decoder=json.loads):
    """Parse argument by reading it is a json string and if that fails as a file path to a json file."""
    is_json = len(arg.split("{")) > 1 or len(arg.split("[")) > 1
    if is_json:
        return json_decoder(arg)

    with open(arg, "r") as json_file:
        json_out = json.load(json_file)
    return json_out


def json_file_or_json_unique_keys(arg):
    """Parse a json file or string and make any duplicate keys unique by appending -i for i from 0 to N"""

    def make_unique(key, dct):
        counter = 0
        unique_key = key

        while unique_key in dct:
            counter += 1
            unique_key = "{}-{}".format(key, counter)
        return unique_key

    def parse_object_pairs(pairs):
        dct = dict()
        for key, value in pairs:
            if key in dct:
                key = make_unique(key, dct)
            dct[key] = value

        return dct

    decoder = JSONDecoder(object_pairs_hook=parse_object_pairs)
    return json_file_or_json(arg, json_decoder=decoder.decode)


def str2bool(arg):
    """Parse a string argument to bool.

    To be used as:
        parser.add_argument('--some_var', type=str2bool, default=False, const=True)

    Arguments parsed to 'True' (case insensitive):
        --some_var true
        --some_var t
        --some_var yes
        --some_var y
        --some_var 1

    Arguments parsed to 'False' (case insensitive):
        --some_var false
        --some_var f
        --some_var no
        --some_var n
        --some_var 0

    See https://stackoverflow.com/a/43357954/4203328
    """
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Could not parse argument {arg} of type {type(arg)}")
