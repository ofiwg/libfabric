#!/usr/bin/env python

import sys
import pprint
import operator
from optparse import OptionParser

try:
	import yaml
except ImportError:
	print ("PyYAML library missing, try: yum install pyyaml")
	sys.exit(1)

class yaml_obj:
	def __init__(self, yaml):
		self.data = yaml

	def __str__(self):
		return pprint.pformat(self.data)

	def __len__(self): return len (self.data)

	def __sub__(self, other):
		assert(len (self.data) == len (other.data))

		res = self.data[0].fromkeys(self.data[0].keys())
		lres = []

		for l1, l2 in zip(self.data, other.data):
			for k in l1.keys():
				if (l1[k] != l2[k]):
					res[k] = l1[k] - l2[k]
				else:
					res[k] = l1[k]
			lres.append(res.copy())

		return yaml_obj(lres)

def main(argv=None):

	parser = OptionParser(description='fabtests yaml parsing utility')
	parser.add_option('-d', action='store_true', default=False, help='difference')
	(options, args) = parser.parse_args()

	if len(args) == 0:
		fd = sys.stdin
	elif len(args) > 1:
		class fd:
			@staticmethod
			def read():
				r1 = map(open, args)
				r2 = map(lambda x: x.read(), r1) 
				return reduce(operator.add, r2)
	else:
		fd = open(args[0], 'r')

	yi = yaml.load_all(fd.read())

	if options.d:
		i = yaml_obj(yi.next())
		j = yaml_obj(yi.next())
		print (j - i)

	for i in yi:
		pprint.pprint(i)

	return 0

if __name__ == "__main__":
	sys.exit(main(sys.argv))
