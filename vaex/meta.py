from __future__ import print_function
__author__ = 'maartenbreddels'
import vaex
import vaex.utils
import logging
import astropy.units
import sys
import yaml

logger = logging.getLogger("vaex.meta")

def main(argv):
	import argparse
	parser = argparse.ArgumentParser(argv[0])
	parser.add_argument('--verbose', '-v', action='count', default=0)
	parser.add_argument('--list', '-l', default=False, action='store_true', help="list columns of input")

	subparsers = parser.add_subparsers(help='type of subtask', dest="task")

	parser_export = subparsers.add_parser('export', help='read meta info')
	parser_export.add_argument("input", help="input dataset")
	parser_export.add_argument('output', help='output file (.yaml or .json)')
	parser_export.add_argument("columns", help="list of columns to export (or all when empty)", nargs="*")
	parser_export.add_argument('--all', dest="all", action='store_true', default=False, help="Also export missing values (useful for having a template)")

	parser_import = subparsers.add_parser('import', help='read meta info')
	parser_import.add_argument('input', help='input meta file (.yaml or .json)')
	parser_import.add_argument("output", help="output dataset")
	parser_import.add_argument("columns", help="list of columns to export (or all when empty)", nargs="*")
	parser_import.add_argument('--overwrite', help="overwrite existing entries", default=False, action='store_true')

	args = parser.parse_args(argv[1:])

	verbosity = ["ERROR", "WARNING", "INFO", "DEBUG"]
	logging.getLogger("vaex").setLevel(verbosity[min(3, args.verbose)])

	if args.task == "export":
		ds = vaex.open(args.input)
		column_names = ds.get_column_names(strings=True, virtual=True)
		if args.all:
			output_data = dict(description=ds.description,
						 	descriptions={name:ds.descriptions.get(name, "") for name in column_names},
							ucds={name:ds.ucds.get(name, "") for name in column_names},
							units={name:str(ds.units.get(name, "")) for name in column_names}, #{name:str(unit) for name, unit in ds.units.items()},
						   )
		else:
			output_data = dict(description=ds.description,
						 	descriptions=ds.descriptions,
							ucds=ds.ucds,
							units={name:str(unit) for name, unit in ds.units.items()},
						   )
		if args.output == "-":
			yaml.safe_dump(output_data, sys.stdout, default_flow_style=False) #, encoding='utf-8',  allow_unicode=True)
		else:
			vaex.utils.write_json_or_yaml(args.output, output_data)
			print("wrote %s" % args.output)
	if args.task == "import":
		if args.input == "-":
			data = yaml.load(sys.stdin)
		else:
			data = vaex.utils.read_json_or_yaml(args.input)

		ds = vaex.open(args.output)

		units = data["units"]
		ucds = data["ucds"]
		for column_name in ds.get_column_names():
			if (args.overwrite or column_name not in ds.units) and column_name in units:
				ds.units[column_name] = astropy.units.Unit(units[column_name])
			if (args.overwrite or column_name not in ds.ucds) and column_name in ucds:
				ds.ucds[column_name] = ucds[column_name]
		ds.write_meta()
		print("updated meta data in %s" % args.output)

if __name__ == "__main__":
	main(sys.argv)

