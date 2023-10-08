from foosball import main, get_argparse

if __name__ == '__main__':
    ap = get_argparse()
    main(vars(ap.parse_args()))
