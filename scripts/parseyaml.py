#!/usr/bin/env python

import sys
import pprint

try:
	import yaml
except ImportError:
	print ("PyYAML library missing, try: yum install pyyaml")
	sys.exit(1)

def main(argv=None):
	if argv is None:
		argv = sys.argv

	if len (sys.argv) == 1:
		fd = sys.stdin
	else:
		fd = open(sys.argv[1], 'r')

	yamlobj = yaml.load_all(fd.read())

	for i in yamlobj:
		pprint.pprint(i)

	return 0

if __name__ == "__main__":
	sys.exit(main(sys.argv))
