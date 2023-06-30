import sys
import os

sys.path.append(os.environ['CLOUDBEES_CONFIG'])

import subprocess
import re
import cloudbees_config
import common
import shlex

# A Jenkins env variable for job name is composed of the name of the jenkins job and the branch name
# it is building for. for e.g. in our case jobname = 'ofi_libfabric/master'
class Test:

    def __init__ (self, jobname, buildno, testname, core_prov, fabric,
                  hosts, ofi_build_mode, user_env, mpitype=None, util_prov=None):
        self.jobname = jobname
        self.buildno = buildno
        self.testname = testname
        self.core_prov = core_prov
        self.util_prov = f'ofi_{util_prov}' if util_prov != None else ''
        self.fabric = fabric
        self.hosts = hosts
        self.mpi_type = mpitype
        self.ofi_build_mode = ofi_build_mode
        if (len(hosts) == 1):
            self.server = hosts[0]
            self.client = hosts[0]
        elif (len(hosts) == 2):
            self.server = hosts[0]
            self.client = hosts[1]

        self.nw_interface = cloudbees_config.interface_map[self.fabric]
        self.libfab_installpath = f'{cloudbees_config.install_dir}/'\
                                  f'{self.jobname}/{self.buildno}/'\
                                  f'{self.ofi_build_mode}'
        if (self.core_prov == 'ucx'):
            self.libfab_installpath += "/ucx"

        self.middlewares_path = f'{cloudbees_config.install_dir}/'\
                                   f'{self.jobname}/{self.buildno}/'\
                                   'middlewares'
        self.ci_logdir_path = f'{cloudbees_config.install_dir}/'\
                                   f'{self.jobname}/{self.buildno}/'\
                                   'log_dir'
        self.env = user_env

        self.mpi = ''
        if (self.mpi_type == 'impi'):
            self.mpi = IMPI(self.core_prov, self.hosts,
                            self.libfab_installpath, self.nw_interface,
                            self.server, self.client, self.env, self.util_prov)
        elif (self.mpi_type == 'ompi'):
            self.mpi = OMPI(self.core_prov, self.hosts,
                             self.libfab_installpath, self.nw_interface,
                             self.server, self.client, self.env,
                             self.middlewares_path, self.util_prov)
        elif (self.mpi_type == 'mpich'):
            self.mpi = MPICH(self.core_prov, self.hosts,
                             self.libfab_installpath, self.nw_interface,
                             self.server, self.client, self.env,
                             self.middlewares_path, self.util_prov)


class FiInfoTest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                     hosts, ofi_build_mode, user_env, None, util_prov)

        self.fi_info_testpath =  f'{self.libfab_installpath}/bin'

    @property
    def cmd(self):
        return f"{self.fi_info_testpath}/fi_info "

    @property
    def options(self):
        if (self.util_prov):
            opts  = f"-f {self.fabric} -p {self.core_prov};{self.util_prov}"
        elif (self.core_prov == 'psm3'):
            opts = f"-p {self.core_prov}"
        else:
            opts = f"-f {self.fabric} -p {self.core_prov}"

        return opts

    def execute_cmd(self):
        command = self.cmd + self.options
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)


class Fabtest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, None, util_prov)
        self.fabtestpath = f'{self.libfab_installpath}/bin'
        self.fabtestconfigpath = f'{self.libfab_installpath}/share/fabtests'

    def get_exclude_file(self):
        path = self.libfab_installpath
        efile_path = f'{path}/share/fabtests/test_configs'

        prov = self.util_prov if self.util_prov else self.core_prov
        efile_old = f'{efile_path}/{prov}/{prov}.exclude'

        if self.util_prov:
            efile = f'{efile_path}/{self.util_prov}/{self.core_prov}/exclude'
        else:
            efile = f'{efile_path}/{self.core_prov}/exclude'

        if os.path.isfile(efile):
            return efile
        elif os.path.isfile(efile_old):
            return efile_old
        else:
            print(f"Exclude file: {efile} not found!")
            return None

    @property
    def cmd(self):
        return f"{self.fabtestpath}/runfabtests.sh "

    @property
    def options(self):
        opts = f"-T 300 -vvv -p {self.fabtestpath} -S "
        if (self.core_prov != 'shm' and self.nw_interface):
            opts += f"-s {common.get_node_name(self.server, self.nw_interface)} "
            opts += f"-c {common.get_node_name(self.client, self.nw_interface)} "

        if (self.core_prov == 'shm'):
            opts += f"-s {self.server} "
            opts += f"-c {self.client} "
            opts += "-N "

        if (self.core_prov == 'ucx'):
            opts += "-b "

        if (self.ofi_build_mode == 'dl'):
            opts += "-t short "
        else:
            opts += "-t all "

        if (self.core_prov == 'sockets' and self.ofi_build_mode == 'reg'):
            complex_test_file = f'{self.libfab_installpath}/share/fabtests/'\
                                f'test_configs/{self.core_prov}/quick.test'
            if (os.path.isfile(complex_test_file)):
                opts += "-u {complex_test_file} "
            else:
                print(f"{self.core_prov} Complex test file not found")

        if (self.ofi_build_mode != 'reg' or self.core_prov == 'udp'):
            opts += "-e \'ubertest,multinode\' "

        efile = self.get_exclude_file()
        if efile:
            opts += "-R "
            opts += f"-f {efile} "

        for key in self.env:
            opts += f"-E {key}={self.env[key]} "

        if self.util_prov:
            opts += f"{self.core_prov};{self.util_prov} "
        else:
            opts += f"{self.core_prov} "

        if (self.core_prov == 'shm'):
            opts += f"{self.server} {self.server} "
        else:
            opts += f"{self.server} {self.client} "

        return opts

    @property
    def execute_condn(self):
        return True

    def execute_cmd(self):
        curdir = os.getcwd()
        os.chdir(self.fabtestconfigpath)
        command = self.cmd + self.options
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)
        os.chdir(curdir)


class ShmemTest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                    hosts, ofi_build_mode, user_env, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                            hosts, ofi_build_mode, user_env, None, util_prov)

        self.n = 4
        self.ppn = 2
        self.shmem_dir = f'{self.middlewares_path}/shmem'
        self.hydra = f'{cloudbees_config.hydra}'
        self.shmem_testname = ''
        self.threshold = '1'
        self.isx_shmem_total_size = 33554432
        self.isx_shmem_kernel_max = 134217728
        self.prk_iterations = 10
        self.prk_first_arr_dim = 1000
        self.prk_second_arr_dim = 1000
        if self.util_prov:
            self.prov = f'{self.core_prov};{self.util_prov}'
        else:
            self.prov = self.core_prov

        self.test_dir = {
            'unit'  : 'SOS',
            'uh'    : 'tests-uh',
            'isx'   : 'ISx/SHMEM',
            'prk'   : 'PRK/SHMEM'
        }

        self.make = {
            'unit'  : 'make VERBOSE=1',
            'uh'    : 'make C_feature_tests-run',
            'isx'   : '',
            'prk'   : ''
        }

        self.shmem_environ = {
            'SHMEM_OFI_USE_PROVIDER': self.prov,
            'OSHRUN_LAUNCHER'		: self.hydra,
            'PATH'					: f'{self.shmem_dir}/bin:$PATH',
            'LD_LIBRARY_PATH'		: f'{self.shmem_dir}/lib:'\
                                        f'{self.libfab_installpath}/lib',
            'SHMEM_SYMMETRIC_SIZE'	: '4G',
            'LD_PRELOAD'			: f'{self.libfab_installpath}'\
                                       '/lib/libfabric.so',
            'threshold'              : self.threshold
        }

    def export_env(self):
        environ = ''
        if self.shmem_testname == 'isx' or self.shmem_testname == 'prk':
            self.threshold = '0'

        for key,val in self.shmem_environ.items():
            environ += f"export {key}={val}; "
        return environ

    def cmd(self):
        cmd = ''
        if self.shmem_testname == 'unit':
            cmd += f"{self.make[self.shmem_testname]} "
            cmd += "mpiexec.hydra "
            cmd += f"-n {self.n} "
            cmd += f"-np {self.ppn} "
            cmd += 'check'
        elif self.shmem_testname == 'uh':
            cmd += f'{self.make[self.shmem_testname]}'
        elif self.shmem_testname == 'isx':
            cmd += f"oshrun -np 4 ./bin/isx.strong {self.isx_shmem_kernel_max}"\
                    " output_strong; "
            cmd += f"oshrun -np 4 ./bin/isx.weak {self.isx_shmem_total_size} "\
                    "output_weak; "
            cmd += f"oshrun -np 4 ./bin/isx.weak_iso "\
                   f"{self.isx_shmem_total_size} output_weak_iso "
        elif self.shmem_testname == 'prk':
            cmd += f"oshrun -np 4 ./Stencil/stencil {self.prk_iterations} "\
                   f"{self.prk_first_arr_dim}; "
            cmd += f"oshrun -np 4 ./Synch_p2p/p2p {self.prk_iterations} "\
                   f"{self.prk_first_arr_dim} {self.prk_second_arr_dim}; "
            cmd += f"oshrun -np 4 ./Transpose/transpose {self.prk_iterations} "\
                   f"{self.prk_first_arr_dim} "

        return cmd


    @property
    def execute_condn(self):
        #make always true when verbs and sockets are passing
        return True if (self.core_prov == 'tcp') \
                    else False

    def execute_cmd(self, shmem_testname):
        self.shmem_testname = shmem_testname
        cwd = os.getcwd()
        os.chdir(f'{self.shmem_dir}/{self.test_dir[self.shmem_testname]}')
        print("Changed directory to "\
              f'{self.shmem_dir}/{self.test_dir[self.shmem_testname]}')
        command = f"bash -c \'{self.export_env()} {self.cmd()}\'"
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)
        os.chdir(cwd)

class MultinodeTests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, None, util_prov)
        self.fabtestpath = f'{self.libfab_installpath}/bin'
        self.fabtestconfigpath = f'{self.libfab_installpath}/share/fabtests'
        self.n = 2
        self.ppn = 64
        self.iterations = 1
        self.method = 'msg'
        self.pattern = "full_mesh"

    @property
    def cmd(self):
        return f"{self.fabtestpath}/runmultinode.sh "

    @property
    def options(self):
        opts = f"-h {common.get_node_name(self.server, self.nw_interface)}"
        opts += f",{common.get_node_name(self.client, self.nw_interface)}"
        opts += f" -n {self.ppn}"
        opts += f" -I {self.iterations}"
        opts += f" -z {self.pattern}"
        opts += f" -C {self.method}"
        if self.util_prov:
            opts += f" -p {self.core_prov};{self.util_prov}"
        else:
            opts += f" -p {self.core_prov}"
        opts += f" --ci {self.fabtestpath}/" #enable ci mode to disable tput

        return opts

    @property
    def execute_condn(self):
        return True

    def execute_cmd(self):
        if self.util_prov:
            prov = f"{self.core_prov}-{self.util_prov} "
        else:
            prov = self.core_prov
        curdir = os.getcwd()
        os.chdir(self.fabtestconfigpath)
        command = self.cmd + self.options
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)
        os.chdir(curdir)

class ZeFabtests(Test):
    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, None, util_prov)

        self.fabtestpath = f'{self.libfab_installpath}/bin'
        self.zefabtest_script_path = f'{cloudbees_config.ze_testpath}'
        self.fabtestconfigpath = f'{self.libfab_installpath}/share/fabtests'

    @property
    def cmd(self):
        return f'{self.zefabtest_script_path}/runfabtests_ze.sh '

    def options(self, test_name):
        opts = f"-p {self.fabtestpath} "
        opts += f"-B {self.fabtestpath} "
        opts += f"-t {test_name} "
        opts += f"{self.server} {self.client} "
        return opts

    @property
    def execute_condn(self):
        return True if (self.core_prov == 'shm') else False

    def execute_cmd(self, test_name):
        curdir = os.getcwd()
        os.chdir(self.fabtestconfigpath)
        command = self.cmd + self.options(test_name)
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)
        os.chdir(curdir)


class OMPI:
    def __init__(self, core_prov, hosts, libfab_installpath, nw_interface,
                 server, client, environ, middlewares_path, util_prov=None):

        self.ompi_src = f'{middlewares_path}/ompi'
        self.core_prov = core_prov
        self.hosts = hosts
        self.util_prov = util_prov
        self.libfab_installpath = libfab_installpath
        self.nw_interface = nw_interface
        self.server = server
        self.client = client
        self.environ = environ
        self.n = 4
        self.ppn = 2

    @property
    def env(self):
        cmd = "bash -c \'"
        if (self.util_prov):
            cmd += f"export FI_PROVIDER={self.core_prov}\\;{self.util_prov}; "
        else:
            cmd += f"export FI_PROVIDER={self.core_prov}; "
        cmd += "export I_MPI_FABRICS=ofi; "
        cmd += f"export LD_LIBRARY_PATH={self.ompi_src}/lib:$LD_LIBRARY_PATH; "
        cmd += f"export LD_LIBRARY_PATH={self.libfab_installpath}/lib/:"\
                "$LD_LIBRARY_PATH; "
        cmd += f"export PATH={self.ompi_src}/bin:$PATH; "
        cmd += f"export PATH={self.libfab_installpath}/bin:$PATH; "
        return cmd

    @property
    def options(self):
        opts = f"-np {self.n} "
        hosts = '\',\''.join([':'.join([common.get_node_name(host, \
                         self.nw_interface), str(self.ppn)]) \
                for host in self.hosts])
        opts += f"--host \'{hosts}\' "
        if self.util_prov:
            opts += f"--mca mtl_ofi_provider_include {self.core_prov}\\;"\
                    f"{self.util_prov} "
            opts += f"--mca btl_ofi_provider_include {self.core_prov}\\;"\
                    f"{self.util_prov} "
        else:
            opts += f"--mca mtl_ofi_provider_include {self.core_prov} "
            opts += f"--mca btl_ofi_provider_include {self.core_prov} "
        opts += "--mca orte_base_help_aggregate 0 "
        # This is necessary to prevent verbs from printing warning messages
        # The test still uses libfabric verbs even when enabled.
        # if (self.core_prov == 'verbs'):
        #     opts += "--mca btl_openib_allow_ib 1 "
        opts += "--mca mtl ofi "
        opts += "--mca pml cm -tag-output "
        for key in self.environ:
            opts += f"-x {key}={self.environ[key]} "

        return opts

    @property
    def cmd(self):
        return f"{self.ompi_src}/bin/mpirun {self.options}"

class MPICH:
    def __init__(self, core_prov, hosts, libfab_installpath, nw_interface,
                 server, client, environ, middlewares_path, util_prov=None):

        self.mpich_src = f'{middlewares_path}/mpich'
        self.core_prov = core_prov
        self.hosts = hosts
        self.util_prov = util_prov
        self.libfab_installpath = libfab_installpath
        self.nw_interface = nw_interface
        self.server = server
        self.client = client
        self.environ = environ
        self.n = 4
        self.ppn = 1

    @property
    def env(self):
        cmd = "bash -c \'"
        if (self.util_prov):
            cmd += f"export FI_PROVIDER={self.core_prov}\\;{self.util_prov}; "
        else:
            cmd += f"export FI_PROVIDER={self.core_prov}; "
        cmd += "export I_MPI_FABRICS=ofi; "
        cmd += "export MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS=0; "
        cmd += "export MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG=1; "
        cmd += f"export LD_LIBRARY_PATH={self.mpich_src}/lib:$LD_LIBRARY_PATH; "
        cmd += f"export LD_LIBRARY_PATH={self.libfab_installpath}/lib/:"\
               "$LD_LIBRARY_PATH; "
        cmd += f"export PATH={self.mpich_src}/bin:$PATH; "
        cmd += f"export PATH={self.libfab_installpath}/bin:$PATH; "
        return cmd

    @property
    def options(self):
        opts = f"-n {self.n} "
        opts += f"-ppn {self.ppn} "
        opts += "-launcher ssh "
        # Removed because sbatch does this for us whenwe use mpirun
        # opts += f"-hosts {common.get_node_name(self.server, self.nw_interface)},"\
        #         f"{common.get_node_name(self.client, self.nw_interface)} "
        for key in self.environ:
            opts += f"-genv {key} {self.environ[key]} "

        return opts

    @property
    def cmd(self):
        return f"{self.mpich_src}/bin/mpirun {self.options}"


class IMPI:
    def __init__(self, core_prov, hosts, libfab_installpath, nw_interface,
                 server, client, environ, util_prov=None):

        self.impi_src = cloudbees_config.impi_root
        self.core_prov = core_prov
        self.hosts = hosts
        self.util_prov = util_prov
        self.libfab_installpath = libfab_installpath
        self.nw_interface = nw_interface
        self.server = server
        self.client = client
        self.environ = environ
        self.n = 4
        self.ppn = 1

    @property
    def env(self):
        cmd = f"bash -c \'source {self.impi_src}/env/vars.sh "\
              "-i_mpi_ofi_internal=0; "
        if (self.util_prov):
            cmd += f"export FI_PROVIDER={self.core_prov}\\;{self.util_prov}; "
        else:
            cmd += f"export FI_PROVIDER={self.core_prov}; "
        cmd += "export I_MPI_FABRICS=ofi; "
        cmd += f"export LD_LIBRARY_PATH={self.impi_src}/lib:$LD_LIBRARY_PATH; "
        cmd += f"export LD_LIBRARY_PATH={self.impi_src}/lib/release:"\
               "$LD_LIBRARY_PATH; "
        cmd += f"export LD_LIBRARY_PATH={self.libfab_installpath}/lib/:"\
               "$LD_LIBRARY_PATH; "
        cmd += f"export PATH={self.libfab_installpath}/bin:$PATH; "
        return cmd

    @property
    def options(self):
        opts = f"-n {self.n} "
        opts += f"-ppn {self.ppn} "
        opts += f"-hosts {common.get_node_name(self.server, self.nw_interface)},"\
                f"{common.get_node_name(self.client, self.nw_interface)} "
        for key in self.environ:
            opts += f"-genv {key} {self.environ[key]} "

        return opts

    @property
    def cmd(self):
        return f"{self.impi_src}/bin/mpiexec {self.options}"


class IMBtests(Test):
    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, mpitype, ofi_build_mode, user_env, test_group,
                 util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, user_env, mpitype,
                         util_prov)

        self.test_group = test_group
        self.mpi_type = mpitype
        self.imb_src = ''
        self.imb_tests = {
                             '1' :[
                                      'MPI1',
                                      'P2P'
                                  ],
                             '2' :[
                                      'EXT',
                                      'IO'
                                  ],
                             '3' :[
                                      'NBC',
                                      'RMA',
                                      'MT'
                                  ]
                         }
        self.iter = 100
        self.include = {
                        'MPI1':[
                                   'Biband',
                                   'Uniband',
                                   'PingPongAnySource',
                                   'PingPingAnySource',
                                   'PingPongSpecificSource',
                                   'PingPingSpecificSource'
                               ],
                        'P2P':[],
                        'EXT':[],
                        'IO':[],
                        'NBC':[],
                        'RMA':[],
                        'MT':[]
                       }
        self.exclude = {
                        'MPI1':[],
                        'P2P':[],
                        'EXT':[],
                        'IO':[],
                        'NBC':[],
                        'RMA':[
                                  'Accumulate',
                                  'Get_accumulate',
                                  'Fetch_and_op',
                                  'Compare_and_swap'
                              ],
                        'MT':[]
                       }
        self.imb_src = f'{self.middlewares_path}/{self.mpi_type}/imb'

    @property
    def execute_condn(self):
        # Mpich and ompi are excluded to save time. Run manually if needed
        return (self.mpi_type == 'impi')

    def imb_cmd(self, imb_test):
        print(f"Running IMB-{imb_test}")
        cmd = f"{self.imb_src}/IMB-{imb_test} "
        if (imb_test != 'MT'):
            cmd += f"-iter {self.iter} "

        if (len(self.include[imb_test]) > 0):
            cmd += f"-include {','.join(self.include[imb_test])}"

        if (len(self.exclude[imb_test]) > 0):
            cmd += f"-exclude {','.join(self.exclude[imb_test])}"

        return cmd

    def execute_cmd(self):
        for test_type in self.imb_tests[self.test_group]:
                outputcmd = shlex.split(self.mpi.env + self.mpi.cmd + \
                                        self.imb_cmd(test_type) + '\'')
                common.run_command(outputcmd)


class OSUtests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, mpitype, ofi_build_mode, user_env, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, user_env, mpitype,
                         util_prov)

        self.n_ppn = {
                          'pt2pt':      (2, 1),
                          'collective': (4, 2),
                          'one-sided':  (2, 1),
                          'startup':    (2, 1)
                     }
        self.osu_src = f'{self.middlewares_path}/{mpitype}/osu/libexec/'\
                       'osu-micro-benchmarks/mpi/'
        self.mpi_type = mpitype

    @property
    def execute_condn(self):
        # mpich-tcp, ompi are the only osu test combinations failing
        return False if ((self.mpi_type == 'mpich' and self.core_prov == 'tcp') or \
                          self.mpi_type == 'ompi') \
                     else True

    def osu_cmd(self, test_type, test):
        print(f"Running OSU-{test_type}-{test}")
        cmd = f'{self.osu_src}/{test_type}/{test} '
        return cmd

    def execute_cmd(self):
        assert(self.osu_src)
        p = re.compile('osu_put*')
        for root, dirs, tests in os.walk(self.osu_src):
            for test in tests:
                self.mpi.n = self.n_ppn[os.path.basename(root)][0]
                self.mpi.ppn = self.n_ppn[os.path.basename(root)][1]

                if (test == 'osu_latency_mp' and self.core_prov == 'verbs'):
                    self.env['IBV_FORK_SAFE'] = '1'

                if(p.search(test) == None):
                    osu_command = self.osu_cmd(os.path.basename(root), test)
                    outputcmd = shlex.split(self.mpi.env + self.mpi.cmd + \
                                            osu_command + '\'')
                    common.run_command(outputcmd)

                if (test == 'osu_latency_mp' and self.core_prov == 'verbs'):
                    self.env.pop('IBV_FORK_SAFE')


class MpichTestSuite(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, mpitype, ofi_build_mode, user_env, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, user_env, mpitype,
                         util_prov)

        self.mpichsuitepath = f'{self.middlewares_path}/{mpitype}/'\
                              'mpichsuite/test/mpi/'
        self.pwd = os.getcwd()
        self.mpi_type = mpitype

    def testgroup(self, testgroupname):
        testpath = f'{self.mpichsuitepath}/{testgroupname}'
        tests = []
        with open(f'{testpath}/testlist') as file:
            for line in file:
                if(line[0] != '#' and  line[0] != '\n'):
                    tests.append((line.rstrip('\n')).split(' '))

        return tests

    def set_options(self, nprocs, timeout=None):
        self.mpi.n = nprocs
        if (timeout != None):
            os.environ['MPIEXEC_TIMEOUT']=timeout


    @property
    def execute_condn(self):
        return (self.mpi_type == 'impi' or \
               (self.mpi_type == 'mpich' and self.core_prov == 'verbs'))

    def execute_cmd(self, testgroupname):
        print("Running Tests: " + testgroupname)
        tests = []
        time = None
        os.chdir(f'{self.mpichsuitepath}/{testgroupname}')
        tests = self.testgroup(testgroupname)
        for test in tests:
            testname = test[0]
            nprocs = test[1]
            args = test[2:]
            for item in args:
               itemlist =  item.split('=')
               if (itemlist[0] == 'timelimit'):
                   time = itemlist[1]
            self.set_options(nprocs, timeout=time)
            testcmd = f'./{testname}'
            outputcmd = shlex.split(self.mpi.env + self.mpi.cmd + testcmd + '\'')
            common.run_command(outputcmd)
        os.chdir(self.pwd)


class OneCCLTests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, None, util_prov)

        self.oneccl_path = f'{self.middlewares_path}/oneccl/'
        self.test_dir = f'{self.middlewares_path}/oneccl/ci_tests'
        if self.util_prov:
            self.prov = f"{self.core_prov}\;{self.util_prov}"
        else:
            self.prov = self.core_prov
        self.oneccl_environ = {
            'FI_PROVIDER'               : f"\"{self.prov}\"",
            'CCL_ATL_TRANSPORT'         : 'ofi',
            'CCL_ATL_TRANSPORT_LIST'    : 'ofi'
        }

        self.ld_library = [
                            f'{self.libfab_installpath}/lib',
                            f'{self.oneccl_path}/build/_install/lib'
        ]

    def export_env(self):
        environ = f"source {cloudbees_config.oneapi_root}/setvars.sh; "
        environ += f"source {self.oneccl_path}/build/_install/env/vars.sh; "
        if self.core_prov == 'psm3':
            self.oneccl_environ['PSM3_MULTI_EP'] = '1'

        for key, val in self.oneccl_environ.items():
            environ += f"export {key}={val}; "

        ld_library_path = 'LD_LIBRARY_PATH='
        for item in self.ld_library:
            ld_library_path += f'{item}:'

        environ += f"export {ld_library_path}$LD_LIBRARY_PATH; "
        return environ

    def cmd(self):
        return './run.sh '

    def options(self):
        opts = "--mode cpu "
        return opts

    @property
    def execute_condn(self):
        return True

    @property
    def execute_condn(self):
        return True

    def execute_cmd(self):
        curr_dir = os.getcwd()
        os.chdir(self.test_dir)
        command = f"bash -c \'{self.export_env()} {self.cmd()} "\
                  f"{self.options()}\'"
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)
        os.chdir(curr_dir)

class OneCCLTestsGPU(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, None, util_prov)

        self.n = 2
        self.ppn = 1
        self.oneccl_path = f'{self.middlewares_path}/oneccl_gpu/build'
        if self.util_prov:
            self.prov = f"{self.core_prov}\;{self.util_prov}"
        else:
            self.prov = self.core_prov

        self.onecclgpu_environ = {
            'FI_PROVIDER'       : self.prov,
            # 'LD_PRELOAD'        : f"{self.libfab_installpath}/lib/libfabric.so",
            'CCL_ATL_TRANSPORT' : 'ofi',
            'CCL_ROOT'          : f"{self.oneccl_path}/_install"
        }

        self.ld_library = [
                            f'{self.libfab_installpath}/lib',
                            '$LD_LIBRARY_PATH',
                            f'{self.oneccl_path}/_install/lib'
        ]

        self.tests = {
            'examples'      : [
                                'sycl_allgatherv_custom_usm_test',
                                'sycl_allgatherv_inplace_test',
                                'sycl_allgatherv_inplace_usm_test',
                                'sycl_allgatherv_test',
                                'sycl_allgatherv_usm_test',
                                'sycl_allreduce_inplace_usm_test',
                                'sycl_allreduce_test',
                                'sycl_allreduce_usm_test',
                                'sycl_alltoall_test',
                                'sycl_alltoall_usm_test',
                                'sycl_alltoallv_test',
                                'sycl_alltoallv_usm_test',
                                'sycl_broadcast_test',
                                'sycl_broadcast_usm_test',
                                'sycl_reduce_inplace_usm_test',
                                'sycl_reduce_scatter_test',
                                'sycl_reduce_scatter_usm_test',
                                'sycl_reduce_test',
                                'sycl_reduce_usm_test'
                            ],
            'functional'    : [
                                'allgatherv_test',
                                'alltoall_test',
                                'alltoallv_test',
                                'bcast_test',
                                'reduce_scatter_test',
                                'reduce_test'
                            ]
            }

    def export_env(self):
        environ = f"source {cloudbees_config.impi_root}/env/vars.sh "\
                   "-i_mpi_internal=0; "
        environ += f"source {cloudbees_config.intel_compiler_root}/env/vars.sh; "
        for key, val in self.onecclgpu_environ.items():
            environ += f"export {key}={val}; "

        ld_library_path = 'LD_LIBRARY_PATH='
        for item in self.ld_library:
            ld_library_path += f'{item}:'

        environ += f"export {ld_library_path}$LD_LIBRARY_PATH; "
        return environ

    def cmd(self):
        return f"{self.oneccl_path}/_install/bin/mpiexec "

    def options(self):
        opts = "-l "
        opts += f"-n {self.n} "
        opts += f"-ppn {self.ppn} "
        opts += f"-hosts {self.server},{self.client} "
        return opts

    @property
    def execute_condn(self):
        return True


    def execute_cmd(self, oneccl_test_gpu):
        curr_dir = os.getcwd()
        if 'examples' in oneccl_test_gpu:
            os.chdir(f"{self.oneccl_path}/_install/examples/sycl")
        else:
            os.chdir(f"{self.oneccl_path}/tests/functional")

        for test in self.tests[oneccl_test_gpu]:
            if '_usm_' in test:
                gpu_selector = 'device'
            else:
                gpu_selector = 'default'

            command = f"bash -c \'{self.export_env()} {self.cmd()} "\
                      f"{self.options()} ./{test} "
            if 'examples' in oneccl_test_gpu:
                command += f"gpu {gpu_selector}"
            command += "\'"

            outputcmd = shlex.split(command)
            common.run_command(outputcmd)
        os.chdir(curr_dir)

class DaosCartTest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, None, util_prov)


        self.set_paths(core_prov)
        print(core_prov)
        self.daos_nodes = cloudbees_config.prov_node_map[core_prov]
        print(self.daos_nodes)
        self.launch_node = self.daos_nodes[0] 

        self.cart_tests = {
                 'corpc_one_node'            :       {'tags' :'cart,corpc,one_node', 'numservers':1, 'numclients':0},
                 'corpc_two_node'            :       {'tags' :'cart,corpc,two_node', 'numservers':2, 'numclients':0},
                 'ctl_one_node'              :       {'tags' :'cart,ctl,one_node', 'numservers':1, 'numclients':1},
                 'ghost_rank_rpc_one_node'   :       {'tags' :'cart,ghost_rank_rpc,one_node', 'numservers':1, 'numclients':0},
                 'group_test'                :       {'tags' :'cart,group_test,one_node', 'numservers':1, 'numclients':0},
                 'iv_one_node'               :       {'tags' :'cart,iv,one_node', 'numservers':1, 'numclients':1},
                 'iv_two_node'               :       {'tags' :'cart,iv,two_node', 'numservers':2, 'numclients':1},
                 'launcher_one_node'         :       {'tags' :'cart,no_pmix_launcher,one_node','numservers':1, 'numclients':1},
                 'multictx_one_node'         :       {'tags' :'cart,no_pmix,one_node', 'numservers':1, 'numclients':0},
                 'rpc_one_node'              :       {'tags' :'cart,rpc,one_node', 'numservers':1, 'numclients':1},
                 'rpc_two_node'              :       {'tags' :'cart,rpc,two_node','numservers':2, 'numclients':1},
                 'swim_notification'         :       {'tags' :'cart,rpc,swim_rank_eviction,one_node', 'numservers':1, 'numclients':1}
        }


    def set_paths(self, core_prov):
        self.ci_middlewares_path = f'{cloudbees_config.build_dir}/{core_prov}'
        self.daos_install_root = f'{self.ci_middlewares_path}/daos/install'
        self.cart_test_scripts = f'{self.daos_install_root}/lib/daos/TESTING/ftest'
        self.mpipath = f'{cloudbees_config.daos_mpi}/bin'
        self.pathlist = [f'{self.daos_install_root}/bin/', self.cart_test_scripts, self.mpipath, \
                       f'{self.daos_install_root}/lib/daos/TESTING/tests']
        self.daos_prereq = f'{self.daos_install_root}/prereq'
        common.run_command(['rm', '-rf', f'{self.ci_middlewares_path}/daos_logs/*'])
        common.run_command(['rm','-rf', f'{self.daos_prereq}/debug/ofi'])
        common.run_command(['ln', '-sfn', self.libfab_installpath, f'{self.daos_prereq}/debug/ofi'])

    @property
    def cmd(self):
        return "python3.6 launch.py "
    
    def remote_launch_cmd(self, testname):

#        The following env variables must be set appropriately prior
#        to running the daos/cart tests OFI_DOMAIN, OFI_INTERFACE, 
#        CRT_PHY_ADDR_STR, PATH, DAOS_TEST_SHARED_DIR DAOS_TEST_LOG_DIR, 
#        LD_LIBRARY_PATH in the script being sourced below.
        launch_cmd = f"ssh {self.launch_node} \"source {self.ci_middlewares_path}/daos_ci_env_setup.sh && \
                           cd {self.cart_test_scripts} &&\" "
        return launch_cmd

    def options(self, testname):
        opts = "-s "
        opts += f"{self.cart_tests[testname]['tags']} "

        if (self.cart_tests[testname]['numservers'] != 0):
            servers = ",".join(self.daos_nodes[:self.cart_tests[testname]['numservers']])
            opts += f"--test_servers={servers} "
        if (self.cart_tests[testname]['numclients'] != 0):
            clients = ",".join(self.daos_nodes[:self.cart_tests[testname]['numclients']])
            opts += f"--test_clients={clients}"
        return opts

    @property
    def execute_condn(self):
        return True
    def execute_cmd(self):
        sys.path.append(f'{self.daos_install_root}/lib64/python3.6/site-packages')
        os.environ['PYTHONPATH']=f'{self.daos_install_root}/lib64/python3.6/site-packages'

        test_dir=self.cart_test_scripts
        curdir=os.getcwd()
        os.chdir(test_dir)
        for test in self.cart_tests:
            print(test)
            command = self.remote_launch_cmd(test) + self.cmd + self.options(test)
            outputcmd = shlex.split(command)
            common.run_command(outputcmd)
            print("--------------------TEST COMPLETED----------------------")
        os.chdir(curdir)
