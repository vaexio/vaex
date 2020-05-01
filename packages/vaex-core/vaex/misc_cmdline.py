def alias_main(argv):
    import argparse
    parser = argparse.ArgumentParser(argv[0])
    # parser.add_argument('--verbose', '-v', action='count', default=0)
    # parser.add_argument('--quiet', '-q', default=False, action='store_true', help="do not output anything")
    # parser.add_argument('--list', '-l', default=False, action='store_true', help="list columns of input")
    # parser.add_argument('--progress', help="show progress (default: %(default)s)", default=True, action='store_true')
    # parser.add_argument('--no-progress', dest="progress", action='store_false')
    # parser.add_argument('--shuffle', "-s", dest="shuffle", action='store_true', default=False)

    subparsers = parser.add_subparsers(help='type of task', dest="task")

    parser_list = subparsers.add_parser('list', help='list aliases')

    parser_add = subparsers.add_parser('add', help='add alias')
    parser_add.add_argument('name', help='name of alias')
    parser_add.add_argument('path', help='path/filename for alias')
    parser.add_argument('-f', '--force', help="force/overwrite existing alias", default=False, action='store_true')

    parser_remove = subparsers.add_parser('remove', help='remove alias')
    parser_remove.add_argument('name', help='name of alias')

    args = parser.parse_args(argv[1:])
    import vaex
    if args.task == "add":
        vaex.aliases[args.name] = args.path
    if args.task == "remove":
        del vaex.aliases[args.name]
    if args.task == "list":
        for name in sorted(vaex.aliases.keys()):
            print("%s: %s" % (name, vaex.aliases[name]))


def make_stat_parser(name):
    import argparse
    parser = argparse.ArgumentParser(name)
    # parser.add_argument('--verbose', '-v', action='count', default=0)
    # parser.add_argument('--quiet', '-q', default=False, action='store_true', help="do not output anything")
    # parser.add_argument('--list', '-l', default=False, action='store_true', help="list columns of input")
    # parser.add_argument('--progress', help="show progress (default: %(default)s)", default=True, action='store_true')
    # parser.add_argument('--no-progress', dest="progress", action='store_false')
    # parser.add_argument('--shuffle', "-s", dest="shuffle", action='store_true', default=False)

    # subparsers = parser.add_subparsers(help='type of task', dest="task")

    # parser_list = subparsers.add_parser('list', help='list aliases')

    # parser = subparsers.add_parser('add', help='add alias')
    parser.add_argument('dataset', help='path or name of dataset')
    parser.add_argument('--fraction', "-f", dest="fraction", type=float, default=1.0, help="fraction of input dataset to export")
    return parser


def stat_main(argv):
    parser = make_stat_parser(argv[0])
    args = parser.parse_args(argv[1:])
    import vaex
    dataset = vaex.open(args.dataset)
    if dataset is None:
        print("Cannot open input: %s" % args.dataset)
        sys.exit(1)
    print("dataset:")
    print("  length: %s" % len(dataset))
    print("  full_length: %s" % dataset.full_length())
    print("  name: %s" % dataset.name)
    print("  path: %s" % dataset.path)
    print("  columns: ")
    desc = dataset.description
    if desc:
        print("    description: %s" % desc)
    for name in dataset.get_column_names():
        print("   - %s: " % name)
        desc = dataset.descriptions.get(name)
        if desc:
            print("  \tdescription: %s" % desc)
        unit = dataset.unit(name)
        if unit:
            print("   \tunit: %s" % unit)
        dtype = dataset.data_type(name)
        print("   \ttype: %s" % dtype.name)
