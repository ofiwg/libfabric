#!/usr/bin/env python3

import argparse, builtins, os, sys, yaml, socket

def parse_args():
	parser = argparse.ArgumentParser(description="libfabric multinode test with slurm")
	parser.add_argument('--dry-run', action='store_true', help='Perform a dry run without making any changes.')
	parser.add_argument("--ci", type=str, help="Commands to prepend to test call. Only used with the internal launcher option", default="")
	parser.add_argument("-x", "--capability", type=str, help="libfabric capability", default="msg")
	parser.add_argument("-i", "--iterations", type=int , help="Number of iterations", default=1)
	parser.add_argument("-l", "--launcher", type=str, choices=['internal', 'srun', 'mpirun'], help="launcher to use for running job. If nothing is specified, test manages processes internally. Available options: internal, srun and mpirun", default="internal")

	required = parser.add_argument_group("Required arguments")
	required.add_argument("-p", "--provider", type=str, help="libfabric provider")
	required.add_argument("-np", "--num-procs", type=int, help="Map process by node, l3cache, etc")
	required.add_argument("-c", "--config", type=str, help="Test configuration")

	srun_required = parser.add_argument_group("Required if using srun")
	srun_required.add_argument("-t", "--procs-per-node", type=int,
							help="Number of procs per node", default=-1)

	args = parser.parse_args()
	return args, parser

def parse_config(config):
	with open(config, "r") as f:
		yconf = yaml.safe_load(f)
	return yconf

def mpi_env(config):
	env = config['environment']
	result = []
	for k in env.keys():
		result.append(f"-x {k}")
	return " ".join(result)

def set_env(config):
	env = config['environment']
	for k, v in env.items():
		os.environ[k] = str(v)

def mpi_mca_params(config):
	try:
		mca = config['mca']
		result = []
		for k, v in env.items():
			result.append(f"--mca {k} {v}")
		return " ".join(result)
	except:
		return ""

def mpi_bind_to(config):
	try:
		return f"--bind-to {config['bind-to']}"
	except:
		return "--bind-to core"

def mpi_map_by(config):
	try:
		return f"--map-by ppr:{config['map-by-count']}:{config['map-by']}"
	except:
		return "--map-by ppr:1:l3"

def execute_cmd(cmd, dry_run):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(script_dir)
	from command import Command
	cmd = Command(cmd, fake=dry_run)
	rc, out = cmd.exec_cmd()
	return rc, out

def split_on_commas(expr): 
	l = []
	c = o = 0 
	s = expr
	stop = False
	while not stop and c != -1 and o != -1:
		o = s.find('[')
		b = s.find(']')
		c = s.find(',')
		while c > o and c < b:
			c = s.find(',', c+1)
		if len(l):
			l.pop()
		if c < o or c > b:
			l += [s[:s.find(',', c)], s[s.find(',', c)+1:]]
			s = l[-1]
		else:
			l += s.split(',')
			stop = True
	for i in range(0, len(l)):
		l[i] = l[i].strip()
	return l

def expand_host_list_sub(expr):
	host_list = []

	# first phase split on the commas first
	open_br = expr.find('[')
	close_br = expr.find(']', open_br)
	if open_br == -1 and close_br == -1:
		return expr.split(',')

	if open_br == -1 or close_br == -1:
		return []

	rangestr = expr[open_br+1 : close_br]

	node = expr[:open_br]

	ranges = rangestr.split(',')

	for r in ranges:
		cur = r.split('-')
		if len(cur) == 2:
			pre = "{:0%dd}" % len(cur[0])
			for idx in range(int(cur[0]), int(cur[1])+1):
				host_list.append(f'{node}{pre.format(idx)}')
		elif len(cur) == 1:
			pre = "{:0%dd}" % len(cur[0])
			host_list.append(f'{node}{pre.format(int(cur[0]))}')

	return host_list

def expand_host_list(expr):
	l = split_on_commas(expr)
	host_list = []
	for e in l:
		host_list += expand_host_list_sub(e)
	return host_list

supported_pm = ['pmi', 'pmi2', 'pmix']

def is_srun_pm_supported():
	rc, out = execute_cmd('srun --mpi=list', False)
	if rc:
		return False
	input_list = out.split('\n')
	cleaned_list = [entry.strip() for entry in input_list if entry.strip()]
	for e in supported_pm:
		if e in cleaned_list[1:]:
			return True
	return False

if __name__ == '__main__':

	# list of providers which do not address specification
	no_addr_prov = ['cxi']

	args, parser = parse_args()

	if not args.config:
		print("**A configuration file is required")
		print(parser.format_help())
		exit(-1)

	mnode = parse_config(args.config)['multinode']
	set_env(mnode)

	if args.launcher == 'srun':
		if not is_srun_pm_supported():
			print(f"**Supported process managers are: {','.join(supported_pm)}")
			print(parser.format_help())
			exit(-1)

	# The script assumes it's already running in a SLURM allocation. It can
	# then srun fi_multinode
	#
	if "pattern" not in mnode:
		print("Test pattern must be defined in the YAML configuration file")
		exit()

	if args.provider in no_addr_prov:
		cmd = f"fi_multinode -n {args.num_procs} -s {socket.gethostname()} " \
			f"-p {args.provider} -x {args.capability} -z {mnode['pattern']} " \
			f"-I {args.iterations} -u {args.launcher.lower()} -T"
	else:
		cmd = f"fi_multinode -n {args.num_procs} -s {socket.gethostname()} " \
			f"-p {args.provider} -x {args.capability} -z '{mnode['pattern']}' " \
			f"-I {args.iterations} -u {args.launcher.lower()} -T"

	if args.launcher.lower() == 'mpirun':
		mpi = f"mpirun {mpi_env(mnode)} {mpi_mca_params(mnode)} {mpi_bind_to(mnode)} "\
			  f"{mpi_map_by(mnode)} -np {args.num_procs} {cmd}"
	elif args.launcher.lower() == 'srun':
		if args.procs_per_node == -1 or args.num_procs == -1:
			print("**Need to specify --procs-per-node and --num-procs")
			print(parser.format_help())
			exit()
		mpi = f"srun --ntasks-per-node {args.procs_per_node} --ntasks {args.num_procs} "\
			  f"{cmd}"
	elif args.launcher.lower() == 'internal':
		if args.procs_per_node == -1:
			print("**Need to specify --procs-per-node")
			print(parser.format_help())
			exit()
		hl = ",".join(expand_host_list(os.environ['SLURM_NODELIST']))
		mpi = f"runmultinode.sh -h {hl} -n {args.procs_per_node} -p {args.provider} " \
			  f"-x {args.capability} -I {args.iterations} -z {mnode['pattern']}"
		if args.ci:
			mpi += f" --ci '{args.ci}'"
	else:
		print("**Unsupported launcher")
		print(parser.format_help())
		exit()

	rc, out = execute_cmd(mpi, args.dry_run)

	print(f"Command completed with {rc}\n{out}")
