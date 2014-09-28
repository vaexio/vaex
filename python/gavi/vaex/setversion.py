#!/usr/bin/env python
import sys
import os
import re

template = """
versionstring = '{versionstring}'
"""
if __name__ == "__main__":
	version = sys.argv[1]
	f = file(os.path.join(os.path.dirname(__file__), "version.py"), "w")
	m = re.match("v([0-9]+)\.([0-9]+)\.([0-9]+)-([0-9]+)-([0-9]+)", version)
	if m is None:
		print "%s is not a valid version string, example: v1.2.16-10-gf6859db, where v1.2.16 should be the tag" % version
		sys.exit(1)
	print m.groups()
	print >>f, template.format(versionstring=version, versionstuple=versionstuple, commits=commits, hash=hash)
	f.close()