#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import re

template = """
versiontring = '{versionstring}'
versiontuple = {versionstuple!r}
commits = {commits}
hash = '{hash}'
"""
if __name__ == "__main__":
	version = sys.argv[1]
	print("deprecated")
	sys.exit(1)
	f = file(os.path.join(os.path.dirname(__file__), "version.py"), "w")
	m = re.match("v([0-9]+)\.([0-9]+)-([0-9]+)-([\w]+)", version)
	if m is None:
		print("%s is not a valid version string, example: v1.2-10-gf6859db, where v1.2.16 should be the tag" % version)
		sys.exit(1)
	groups = m.groups()
	commits = groups[2]
	versionstuple =  list(map(int, groups[:2]))
	versionstuple.append(int(commits))
	versionstuple = tuple(versionstuple)
	hash = groups[3]
	version = ".".join(map(str,versionstuple)) + "-" + str(hash)
	print(template.format(versionstring=version, versionstuple=versionstuple, commits=commits, hash=hash), file=f)
	f.close()