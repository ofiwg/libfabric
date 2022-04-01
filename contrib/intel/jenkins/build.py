import os
import sys

# add jenkins config location to PATH
sys.path.append(os.environ['CI_SITE_CONFIG'])

import ci_site_config
import argparse
import subprocess
import shlex
import common
import re
import shutil


def build_libfabric(libfab_install_path, mode):

    if (os.path.exists(libfab_install_path) != True):
        os.makedirs(libfab_install_path)

    config_cmd = ['./configure', '--prefix={}'.format(libfab_install_path)]
    enable_prov_val = 'yes'

    if (mode == 'dbg'):
        config_cmd.append('--enable-debug')
    elif (mode == 'dl'):
        enable_prov_val='dl'

    for prov in common.enabled_prov_list:
        config_cmd.append('--enable-{}={}'.format(prov, enable_prov_val))
    for prov in common.disabled_prov_list:
         config_cmd.append('--enable-{}=no'.format(prov))

    config_cmd.append('--disable-opx') # we do not test opx in intel jenkins ci
    config_cmd.append('--disable-efa') # we do not test efa in intel jenkins ci

    config_cmd.append('--enable-ze-dlopen')

    common.run_command(['./autogen.sh'])
    common.run_command(shlex.split(" ".join(config_cmd)))
    common.run_command(['make','clean'])
    common.run_command(['make', '-j32'])
    common.run_command(['make','install'])


def build_fabtests(libfab_install_path, mode):

    os.chdir('{}/fabtests'.format(workspace))
    if (mode == 'dbg'):
        config_cmd = ['./configure', '--enable-debug', '--prefix={}' \
                      .format(libfab_install_path),'--with-libfabric={}' \
                      .format(libfab_install_path)]
    else:
        config_cmd = ['./configure', '--prefix={}'.format(libfab_install_path),
                      '--with-libfabric={}'.format(libfab_install_path)]

    common.run_command(['./autogen.sh'])
    common.run_command(config_cmd)
    common.run_command(['make','clean'])
    common.run_command(['make', '-j32'])
    common.run_command(['make', 'install'])

def copy_build_dir(install_path):
    shutil.copytree(ci_site_config.build_dir,
                    '{}/ci_middlewares'.format(install_path))

def skip(install_path):
    if os.getenv('CHANGE_TARGET') is not None:
        change_target = os.environ['CHANGE_TARGET']
    else:
        change_target = 'main'

    command = [
                  '{}/skip.sh'.format(ci_site_config.testpath),
                  '{}'.format(os.environ['WORKSPACE']),
                  '{}'.format(change_target)
              ]
    common.run_command(command)

if __name__ == "__main__":
#read Jenkins environment variables
    # In Jenkins,  JOB_NAME  = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
    # job name is better to use to distinguish between builds of different
    # jobs but with same branch name.
    jobname = os.environ['JOB_NAME']
    buildno = os.environ['BUILD_NUMBER']
    workspace = os.environ['WORKSPACE']

    parser = argparse.ArgumentParser()
    parser.add_argument('--build_item', help="build libfabric or fabtests",
                         choices=['libfabric', 'fabtests', 'builddir', 'skip'])
    parser.add_argument('--ofi_build_mode', help="select buildmode debug or dl", \
                        choices=['dbg', 'dl'])

    args = parser.parse_args()
    build_item = args.build_item

    if (args.ofi_build_mode):
        ofi_build_mode = args.ofi_build_mode
    else:
        ofi_build_mode = 'reg'

    ci_middlewares_install_path = '{installdir}/{jbname}/{bno}' \
                                .format(installdir=ci_site_config.install_dir, \
                                jbname=jobname, bno=buildno)
    install_path = '{installdir}/{jbname}/{bno}/{bmode}' \
                   .format(installdir=ci_site_config.install_dir,
                   jbname=jobname, bno=buildno,bmode=ofi_build_mode)

    p = re.compile('mpi*')

    if (build_item == 'libfabric'):
        build_libfabric(install_path, ofi_build_mode)

    elif (build_item == 'fabtests'):
        build_fabtests(install_path, ofi_build_mode)

    elif (build_item == 'builddir'):
        copy_build_dir(ci_middlewares_install_path)

    elif (build_item == 'skip'):
        skip(ci_middlewares_install_path)

