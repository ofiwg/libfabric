import sys
import os

print(os.environ['CI_SITE_CONFIG'])
sys.path.append(os.environ['CI_SITE_CONFIG'])

import subprocess
import re
import ci_site_config
import common
import shlex

# A Jenkins env variable for job name is composed of the name of the jenkins job and the branch name
# it is building for. for e.g. in our case jobname = 'ofi_libfabric/master'
class Test:

    def __init__ (self, jobname, buildno, testname, core_prov, fabric,
                  hosts, ofi_build_mode, user_env, run_test, mpitype=None, util_prov=None):
        self.jobname = jobname
        self.buildno = buildno
        self.testname = testname
        self.core_prov = core_prov
        self.util_prov = f'ofi_{util_prov}' if util_prov != None else ''
        self.fabric = fabric
        self.hosts = hosts
        self.run_test = run_test
        self.mpi_type = mpitype
        self.ofi_build_mode = ofi_build_mode
        if (len(hosts) == 2):
            self.server = hosts[0]
            self.client = hosts[1]

        self.nw_interface = ci_site_config.interface_map[self.fabric]
        self.libfab_installpath = f'{ci_site_config.install_dir}/'\
                                  f'{self.jobname}/{self.buildno}/'\
                                  f'{self.ofi_build_mode}'
        self.ci_middlewares_path = f'{ci_site_config.install_dir}/'\
                                   f'{self.jobname}/{self.buildno}/'\
                                   'ci_middlewares'
        self.ci_logdir_path = f'{ci_site_config.install_dir}/'\
                                   f'{self.jobname}/{self.buildno}/'\
                                   'log_dir'
        self.env = eval(user_env)

        self.mpi = ''
        if (self.mpi_type == 'impi'):
            self.mpi = IMPI(self.core_prov, self.hosts,
                            self.libfab_installpath, self.nw_interface,
                            self.server, self.client, self.env, self.util_prov)
        elif (self.mpi_type == 'ompi'):
            self.mpi = OMPI(self.core_prov, self.hosts,
                             self.libfab_installpath, self.nw_interface,
                             self.server, self.client, self.env,
                             self.ci_middlewares_path, self.util_prov)
        elif (self.mpi_type == 'mpich'):
            self.mpi = MPICH(self.core_prov, self.hosts,
                             self.libfab_installpath, self.nw_interface,
                             self.server, self.client, self.env,
                             self.ci_middlewares_path, self.util_prov)


class FiInfoTest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, run_test, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                     hosts, ofi_build_mode, user_env, run_test, None, util_prov)

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
        common.run_command(outputcmd, self.ci_logdir_path, self.run_test,
                           self.ofi_build_mode)


class Fabtest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, run_test, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, run_test, None, util_prov)
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
        common.run_command(outputcmd, self.ci_logdir_path, self.run_test,
                self.ofi_build_mode)
        os.chdir(curdir)


class ShmemTest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, run_test, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, run_test, None, util_prov)

        #self.n - number of hosts * number of processes per host
        self.n = 4
        # self.ppn - number of processes per node.
        self.ppn = 2
        self.shmem_dir = f'{self.ci_middlewares_path}/shmem'

    @property
    def cmd(self):
        return f"{ci_site_config.testpath}/run_shmem.sh "

    def options(self, shmem_testname):

        if self.util_prov:
            prov = f"{self.core_prov};{self.util_prov} "
        else:
            prov = self.core_prov

        opts = f"-n {self.n} "
        opts += f"-hosts {self.server},{self.client} "
        opts += f"-shmem_dir={self.shmem_dir} "
        opts += f"-libfabric_path={self.libfab_installpath}/lib "
        opts += f"-prov {prov} "
        opts += f"-test {shmem_testname} "
        opts += f"-server {self.server} "
        opts += f"-inf {ci_site_config.interface_map[self.fabric]}"
        return opts

    @property
    def execute_condn(self):
        #make always true when verbs and sockets are passing
        return True if (self.core_prov == 'tcp') \
                    else False

    def execute_cmd(self, shmem_testname):
        command = self.cmd + self.options(shmem_testname)
        outputcmd = shlex.split(command)
        common.run_command(outputcmd, self.ci_logdir_path,
                           f'{shmem_testname}_{self.run_test}',
                           self.ofi_build_mode)

class MultinodeTests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, run_test, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, run_test, None, util_prov)
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
        common.run_command(outputcmd, self.ci_logdir_path, prov,
                           self.ofi_build_mode)
        os.chdir(curdir)

class ZeFabtests(Test):
    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, run_test, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, run_test, None, util_prov)

        self.fabtestpath = f'{self.libfab_installpath}/bin'
        self.zefabtest_script_path = f'{ci_site_config.ze_testpath}'
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
        common.run_command(outputcmd, self.ci_logdir_path,
                           f'{test_name}', self.ofi_build_mode)
        os.chdir(curdir)


class OMPI:
    def __init__(self, core_prov, hosts, libfab_installpath, nw_interface,
                 server, client, environ, ci_middlewares_path, util_prov=None):

        self.ompi_src = f'{ci_middlewares_path}/ompi'
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
                 server, client, environ, ci_middlewares_path, util_prov=None):

        self.mpich_src = f'{ci_middlewares_path}/mpich'
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
        opts += f"-hosts {common.get_node_name(self.server, self.nw_interface)},"\
                f"{common.get_node_name(self.client, self.nw_interface)} "
        for key in self.environ:
            opts += f"-genv {key} {self.environ[key]} "

        return opts

    @property
    def cmd(self):
        return f"{self.mpich_src}/bin/mpirun {self.options}"


class IMPI:
    def __init__(self, core_prov, hosts, libfab_installpath, nw_interface,
                 server, client, environ, util_prov=None):

        self.impi_src = ci_site_config.impi_root
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
                 hosts, mpitype, ofi_build_mode, user_env, run_test, test_group,
                 util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, user_env, run_test, mpitype,
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
        if (self.mpi_type == 'impi'):
            self.imb_src = ci_site_config.impi_root
        elif (self.mpi_type == 'ompi' or self.mpi_type == 'mpich'):
            self.imb_src = f'{self.ci_middlewares_path}/{self.mpi_type}/imb'

    @property
    def execute_condn(self):
        # Mpich and ompi are excluded to save time. Run manually if needed
        return (self.mpi_type == 'impi' and self.core_prov != 'net')

    def imb_cmd(self, imb_test):
        print(f"Running IMB-{imb_test}")
        cmd = f"{self.imb_src}/bin/IMB-{imb_test} "
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
                common.run_command(outputcmd, self.ci_logdir_path,
                                   f'{self.mpi_type}_{self.run_test}',
                                   self.ofi_build_mode)


class OSUtests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, mpitype, ofi_build_mode, user_env, run_test, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, user_env, run_test, mpitype,
                         util_prov)

        self.n_ppn = {
                          'pt2pt':      (2, 1),
                          'collective': (4, 2),
                          'one-sided':  (2, 1),
                          'startup':    (2, 1)
                     }
        self.osu_src = f'{self.ci_middlewares_path}/{mpitype}/osu/libexec/'\
                       'osu-micro-benchmarks/mpi/'
        self.mpi_type = mpitype

    @property
    def execute_condn(self):
        # mpich-tcp, ompi, and net are the only osu test combinations failing
        return False if ((self.mpi_type == 'mpich' and self.core_prov == 'tcp') or \
                          self.mpi_type == 'ompi' or self.core_prov == 'net') \
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
                    common.run_command(outputcmd, self.ci_logdir_path,
                                       f'{self.mpi_type}_{self.run_test}',
                                       self.ofi_build_mode)

                if (test == 'osu_latency_mp' and self.core_prov == 'verbs'):
                    self.env.pop('IBV_FORK_SAFE')


class MpichTestSuite(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, mpitype, ofi_build_mode, user_env, run_test, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, user_env, run_test, mpitype,
                         util_prov)

        self.mpichsuitepath = f'{self.ci_middlewares_path}/{mpitype}/'\
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
        # net provider shouldn't run with MPI for now
        return ((self.mpi_type == 'impi' and self.core_prov != 'net')
                or (self.mpi_type == 'mpich' and self.core_prov == 'verbs'))

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
            if self.util_prov:
                util_prov = self.util_prov.strip('ofi_')
                log_file_name = f'{self.core_prov}-{util_prov}_' \
                                f'{self.mpi_type}_{self.run_test}'
            else:
                log_file_name = f'{self.core_prov}_{self.mpi_type}_{self.run_test}'

            common.run_command(outputcmd, self.ci_logdir_path, log_file_name,
                                self.ofi_build_mode)
        os.chdir(self.pwd)


class OneCCLTests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, run_test, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, run_test, None, util_prov)

        self.n = 2
        self.ppn = 1
        self.oneccl_path = f'{self.ci_middlewares_path}/oneccl/build'

        self.examples_tests = {
                                  'allgatherv',
                                  'allreduce',
                                  'alltoallv',
                                  'broadcast',
                                  'communicator',
                                  'cpu_allgatherv_test',
                                  'cpu_allreduce_bf16_test',
                                  'cpu_allreduce_test',
                                  'custom_allreduce',
                                  'datatype',
                                  'external_kvs',
                                  'priority_allreduce',
                                  'reduce',
                                  'reduce_scatter',
                                  'unordered_allreduce'
                              }
        self.functional_tests = {
                                    'allgatherv_test',
                                    'allreduce_test',
                                    'alltoall_test',
                                    'alltoallv_test',
                                    'bcast_test',
                                    'reduce_scatter_test',
                                    'reduce_test'
                                }

    @property
    def cmd(self):
        return f"{ci_site_config.testpath}/run_oneccl.sh "

    def options(self, oneccl_test):
        opts = f"-n {self.n} "
        opts += f"-ppn {self.ppn} "
        opts += f"-hosts {self.server},{self.client} "
        opts += f"-prov '{self.core_prov}' "
        opts += f"-test {oneccl_test} "
        opts += f"-libfabric_path={self.libfab_installpath}/lib "
        opts += f'-oneccl_root={self.oneccl_path}'
        return opts

    @property
    def execute_condn(self):
        return True


    def execute_cmd(self, oneccl_test):
        if oneccl_test == 'examples':
                for test in self.examples_tests:
                        command = self.cmd + self.options(oneccl_test) + \
                                  f" {test}"
                        outputcmd = shlex.split(command)
                        common.run_command(outputcmd, self.ci_logdir_path, self.run_test,
                                           self.ofi_build_mode)
        elif oneccl_test == 'functional':
                for test in self.functional_tests:
                        command = self.cmd + self.options(oneccl_test) + \
                                  f" {test}"
                        outputcmd = shlex.split(command)
                        common.run_command(outputcmd, self.ci_logdir_path, self.run_test,
                                           self.ofi_build_mode)

class OneCCLTestsGPU(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, run_test, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, run_test, None, util_prov)

        self.n = 2
        self.ppn = 4
        self.oneccl_path = f'{self.ci_middlewares_path}/oneccl_gpu/build'

        self.examples_tests = {
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
                              }
        self.functional_tests = {
                                    'allgatherv_test',
                                    'alltoall_test',
                                    'alltoallv_test',
                                    'bcast_test',
                                    'reduce_scatter_test',
                                    'reduce_test'
                                }

    @property
    def cmd(self):
        return f"{ci_site_config.testpath}/run_oneccl_gpu.sh "

    def options(self, oneccl_test_gpu):
        opts = f"-n {self.n} "
        opts += f"-ppn {self.ppn} "
        opts += f"-hosts {self.server},{self.client} "
        opts += f"-test {oneccl_test_gpu} "
        opts += f"-libfabric_path={self.libfab_installpath}/lib "
        opts += f'-oneccl_root={self.oneccl_path}'
        return opts

    @property
    def execute_condn(self):
        return True


    def execute_cmd(self, oneccl_test_gpu):
        if oneccl_test_gpu == 'examples':
                for test in self.examples_tests:
                        command = self.cmd + self.options(oneccl_test_gpu) + \
                                  f" {test}"
                        outputcmd = shlex.split(command)
                        common.run_command(outputcmd, self.ci_logdir_path,
                                           self.run_test, self.ofi_build_mode)
        elif oneccl_test_gpu == 'functional':
                for test in self.functional_tests:
                        command = self.cmd + self.options(oneccl_test_gpu) + \
                                  f" {test}"
                        outputcmd = shlex.split(command)
                        common.run_command(outputcmd, self.ci_logdir_path,
                                           self.run_test, self.ofi_build_mode)

class DaosCartTest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, user_env, run_test, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, user_env, run_test, None, util_prov)

        self.set_paths()
        self.set_environment(core_prov,util_prov)
        print(core_prov)
        self.daos_nodes = ci_site_config.prov_node_map[core_prov]
        print(self.daos_nodes)

        self.cart_tests = {
                 'corpc_one_node'            :       {'tags' :'cart,corpc,one_node', 'numservers':1, 'numclients':0},
                 'corpc_two_node'            :       {'tags' :'cart,corpc,two_node', 'numservers':2, 'numclients':0},
                 'ctl_one_node'              :       {'tags' :'cart,ctl,one_node', 'numservers':1, 'numclients':1},
#                 'ghost_rank_rpc_one_node'   :       {'tags' :'cart,ghost_rank_rpc,one_node', 'numservers':1, 'numclients':0},
                 'group_test'                :       {'tags' :'cart,group_test,one_node', 'numservers':1, 'numclients':0},
                 'iv_one_node'               :       {'tags' :'cart,iv,one_node', 'numservers':1, 'numclients':1},
                 'iv_two_node'               :       {'tags' :'cart,iv,two_node', 'numservers':2, 'numclients':1},
                 'launcher_one_node'         :       {'tags' :'cart,no_pmix_launcher,one_node','numservers':1, 'numclients':1},
#                 'multictx_one_node'         :       {'tags' :'cart,no_pmix,one_node', 'numservers':1, 'numclients':0},
                 'rpc_one_node'              :       {'tags' :'cart,rpc,one_node', 'numservers':1, 'numclients':1},
                 'rpc_two_node'              :       {'tags' :'cart,rpc,two_node','numservers':2, 'numclients':1},
                 'swim_notification'         :       {'tags' :'cart,rpc,swim_rank_eviction,one_node', 'numservers':1, 'numclients':1}
        }


    def set_paths(self):
        self.ci_middlewares_path = f'{ci_site_config.ci_middlewares}'
        self.daos_install_root = f'{self.ci_middlewares_path}/daos/install'
        self.cart_test_scripts = f'{self.daos_install_root}/lib/daos/TESTING/ftest'
        self.mpipath = f'{ci_site_config.daos_mpi}/bin'
        self.pathlist = [f'{self.daos_install_root}/bin/', self.cart_test_scripts, self.mpipath, \
                       f'{self.daos_install_root}/lib/daos/TESTING/tests']
        self.daos_prereq = f'{self.daos_install_root}/prereq'
        common.run_command(['rm','-rf', f'{self.daos_prereq}/debug/ofi'])
        common.run_command(['ln', '-sfn', self.libfab_installpath, f'{self.daos_prereq}/debug/ofi'])

    def set_environment(self, core_prov, util_prov):
        prov_name = f'ofi+{core_prov}'
        if util_prov:
            prov_name = f'{prov_name};ofi_{util_prov}'
        if (core_prov == 'verbs'):
            os.environ["OFI_DOMAIN"] = 'mlx5_0'
        else:
            os.environ["OFI_DOMAIN"] = 'ib0'
        os.environ["OFI_INTERFACE"] = 'ib0'
        os.environ["CRT_PHY_ADDR_STR"] = prov_name
        os.environ["PATH"] += os.pathsep + os.pathsep.join(self.pathlist)
        os.environ["DAOS_TEST_SHARED_DIR"] = ci_site_config.daos_share
        os.environ["DAOS_TEST_LOG_DIR"] = ci_site_config.daos_logs
        os.environ["LD_LIBRARY_PATH"] = f'{self.ci_middlewares_path}/daos/install/lib64:{self.mpipath}'

    @property
    def cmd(self):
        return "./launch.py "

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
        print("PATH:" +  os.environ["PATH"])
        print("LD_LIBRARY_PATH:" + os.environ["LD_LIBRARY_PATH"])
        print("MODULEPATH:" +  os.environ["MODULEPATH"])

        test_dir=self.cart_test_scripts
        curdir=os.getcwd()
        os.chdir(test_dir)
        for test in self.cart_tests:
            print(test)
            command = self.cmd + self.options(test)
            outputcmd = shlex.split(command)
            common.run_command(outputcmd, self.ci_logdir_path,
                               self.run_test, self.ofi_build_mode)
            print("--------------------TEST COMPLETED----------------------")
        os.chdir(curdir)
