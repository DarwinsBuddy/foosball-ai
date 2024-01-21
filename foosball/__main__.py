import json

from foosball import main, get_argparse


def from_file(config_file_path):
    with open(config_file_path) as f:
        return json.load(f)


if __name__ == '__main__':
    ap = get_argparse()
    namespace = ap.parse_args()
    config_path = vars(namespace)['config']
    if config_path:
        defaults = from_file(config_path)
        ap.set_defaults(**defaults)
    main(vars(ap.parse_args()))
