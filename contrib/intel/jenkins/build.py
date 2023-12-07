import os
import sys

# add jenkins config location to PATH
sys.path.append(f"{os.environ['WORKSPACE']}/ci_resources/configs/{os.environ['CLUSTER']}")
import cloudbees_config

import argparse
import subprocess
import shlex
import common
import re
import shutil

def build_libfabric(libfab_install_path, mode, hw_type, gpu=False):

	if (os.path.exists(libfab_install_path) != True):
		os.makedirs(libfab_install_path)

	config_cmd = ['./configure', f'--prefix={libfab_install_path}']
	enable_prov_val = 'yes'

	if (mode == 'dbg'):
		config_cmd.append('--enable-debug')
	elif (mode == 'dl'):
		enable_prov_val = 'dl'

	for prov in common.providers[hw_type]['enable']:
		config_cmd.append(f'--enable-{prov}={enable_prov_val}')

	for prov in common.providers[hw_type]['disable']:
		config_cmd.append(f'--enable-{prov}=no')

	for op in common.common_disable_list:
		config_cmd.append(f'--enable-{op}=no')

	if (gpu):
		config_cmd.append('--enable-ze-dlopen')

	common.run_command(['./autogen.sh'])
	common.run_command(shlex.split(" ".join(config_cmd)))
	common.run_command(['make','clean'])
	common.run_command(['make', '-j32'])
	common.run_command(['make','install'])


def build_fabtests(libfab_install_path, mode):
	if (mode == 'dbg'):
		config_cmd = ['./configure', '--enable-debug',
					  f'--prefix={libfab_install_path}',
					  f'--with-libfabric={libfab_install_path}']
	else:
		config_cmd = ['./configure', f'--prefix={libfab_install_path}',
					  f'--with-libfabric={libfab_install_path}']

	common.run_command(['./autogen.sh'])
	common.run_command(config_cmd)
	common.run_command(['make','clean'])
	common.run_command(['make', '-j32'])
	common.run_command(['make', 'install'])

def build_mpich(install_path, libfab_installpath, hw_type):
	mpich_build_dir = f'{install_path}/middlewares/mpich_{hw_type}/mpich'
	cwd = os.getcwd()
	if (os.path.exists(mpich_build_dir)):
		print("configure mpich")
		os.chdir(mpich_build_dir)
		configure_cmd = f"./configure "
		configure_cmd += f"--prefix={install_path}/middlewares/mpich_{hw_type} "
		configure_cmd += f"--with-libfabric={libfab_installpath} "
		configure_cmd += "--disable-oshmem "
		configure_cmd += "--disable-fortran "
		configure_cmd += "--without-ch4-shmmods "
		configure_cmd += "--with-device=ch4:ofi "
		configure_cmd += "--without-ze "
		print(configure_cmd)
		common.run_command(['./autogen.sh'])
		common.run_command(shlex.split(configure_cmd))
		common.run_command(['make','-j'])
		common.run_command(['make','install'])
		os.chdir(cwd)

def build_mpich_osu(install_path, libfab_installpath, hw_type):
	mpich_build = f'{install_path}/middlewares/mpich_{hw_type}'
	osu_build_dir = f'{install_path}/middlewares/mpich_{hw_type}/osu_source'
	cwd = os.getcwd()
	if (os.path.exists(osu_build_dir)):
		os.chdir(osu_build_dir)
		if 'LD_LIBRARY_PATH' in dict(os.environ).keys():
			ld_library_path = os.environ['LD_LIBRARY_PATH']
		else:
			ld_library_path = ''

		if 'PATH' in dict(os.environ).keys():
			path = os.environ['PATH']
		else:
			path = ''

		os.environ['CC']=f'{mpich_build}/bin/mpicc'
		os.environ['CXX']=f'{mpich_build}/bin/mpicxx'
		os.environ['CFLAGS']=f'-I{osu_build_dir}/util'
		os.environ['PATH']=f'{libfab_installpath}/bin:{mpich_build}/bin/:{path}'
		os.environ['LD_LIBRARY_PATH']=f'{libfab_installpath}/lib:'\
									  f'{mpich_build}/bin/lib:{ld_library_path}'
		configure_cmd = f"./configure "
		configure_cmd += f"--prefix={mpich_build}/osu "
		print(f"Building OSU Tests: {configure_cmd}")
		common.run_command(shlex.split(configure_cmd))
		common.run_command(shlex.split("make -j install"))
		os.chdir(cwd)
		os.environ['PATH'] = path
		os.environ['LD_LIBRARY_PATH'] = ld_library_path


def copy_build_dir(install_path):
	middlewares_path = f'{install_path}/middlewares'
	if (os.path.exists(middlewares_path) != True):
		os.makedirs(f'{install_path}/middlewares')

	shutil.copytree(f'{cloudbees_config.build_dir}/shmem',
					f'{middlewares_path}/shmem')
	shutil.copytree(f'{cloudbees_config.build_dir}/oneccl',
					f'{middlewares_path}/oneccl')
	shutil.copytree(f'{cloudbees_config.build_dir}/mpich_water',
					f'{middlewares_path}/mpich_water')
	shutil.copytree(f'{cloudbees_config.build_dir}/mpich_grass',
					f'{middlewares_path}/mpich_grass')

	os.symlink(f'{cloudbees_config.build_dir}/impi',
			   f'{middlewares_path}/impi')
	os.symlink(f'{cloudbees_config.build_dir}/ompi',
			   f'{middlewares_path}/ompi')
	os.symlink(f'{cloudbees_config.build_dir}/oneccl_gpu',
			   f'{middlewares_path}/oneccl_gpu')

def copy_file(file_name):
	if (os.path.exists(f'{workspace}/{file_name}')):
		shutil.copyfile(f'{workspace}/{file_name}',
						f'{install_path}/log_dir/{file_name}')

def log_dir(install_path, release=False):
	if (os.path.exists(f'{install_path}/log_dir') != True):
		os.makedirs(f'{install_path}/log_dir')

	if (release):
		copy_file('Makefile.am.diff')
		copy_file('configure.ac.diff')
		copy_file('release_num.txt')

if __name__ == "__main__":
#read Jenkins environment variables
	# In Jenkins,  JOB_NAME  = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
	# job name is better to use to distinguish between builds of different
	# jobs but with same branch name.
	jobname = os.environ['JOB_NAME']
	buildno = os.environ['BUILD_NUMBER']
	workspace = os.environ['WORKSPACE']
	custom_workspace = os.environ['CUSTOM_WORKSPACE']

	parser = argparse.ArgumentParser()
	parser.add_argument('--build_item', help="build libfabric or fabtests", \
						choices=['libfabric', 'fabtests', 'builddir', 'logdir',\
								 'mpich'])
	parser.add_argument('--build_hw', help="HW type for build",
						choices=['water', 'grass', 'fire', 'electric', 'ucx',
								 'daos', 'gpu'])
	parser.add_argument('--ofi_build_mode', help="select buildmode libfabric "\
						"build mode", choices=['reg', 'dbg', 'dl'])
	parser.add_argument('--build_loc', help="build location for libfabric "\
						"and fabtests", type=str, default='./')
	parser.add_argument('--release', help="This job is likely testing a "\
						"release and will be checked into a git tree.",
						action='store_true')
	parser.add_argument('--gpu', help="Enable ZE dlopen", action='store_true')

	args = parser.parse_args()
	build_item = args.build_item
	build_hw = args.build_hw
	build_loc = args.build_loc
	release = args.release
	gpu = args.gpu

	if (args.ofi_build_mode):
		ofi_build_mode = args.ofi_build_mode
	else:
		ofi_build_mode = 'reg'

	libfab_install_path = f'{custom_workspace}/{build_hw}/{ofi_build_mode}'

	p = re.compile('mpi*')

	curr_dir = os.getcwd()
	os.chdir(build_loc)

	if (build_item == 'libfabric'):
		build_libfabric(libfab_install_path, ofi_build_mode, build_hw, gpu)
	elif (build_item == 'fabtests'):
		build_fabtests(libfab_install_path, ofi_build_mode)
	elif (build_item == 'builddir'):
		copy_build_dir(custom_workspace)
	elif (build_item == 'logdir'):
		log_dir(custom_workspace, release)
	elif(build_item == 'mpich'):
		build_mpich(custom_workspace, libfab_install_path, build_hw)
		build_mpich_osu(custom_workspace, libfab_install_path, build_hw)

	os.chdir(curr_dir)
