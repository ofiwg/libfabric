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
                  hosts, ofi_build_mode, util_prov=None):
        self.jobname = jobname
        self.buildno = buildno
        self.testname = testname
        self.core_prov = core_prov
        self.util_prov = 'ofi_{}'.format(util_prov) if util_prov != None else ''
        self.fabric = fabric
        self.hosts = hosts
        self.ofi_build_mode = ofi_build_mode
        if (len(hosts) == 2):
            self.server = hosts[0]
            self.client = hosts[1]

        self.nw_interface = ci_site_config.interface_map[self.fabric]
        self.libfab_installpath = '{}/{}/{}/{}' \
                                  .format(ci_site_config.install_dir,
                                  self.jobname, self.buildno,
                                  self.ofi_build_mode)
        self.ci_middlewares_path = '{}/{}/{}/ci_middlewares' \
                                   .format(ci_site_config.install_dir,
                                   self.jobname, self.buildno)

        self.env = [('FI_VERBS_MR_CACHE_ENABLE', '1'),\
                    ('FI_VERBS_INLINE_SIZE', '256')] \
                    if self.core_prov == 'verbs' else []


class FiInfoTest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                     hosts, ofi_build_mode, util_prov)

        self.fi_info_testpath =  '{}/bin'.format(self.libfab_installpath)

    @property
    def cmd(self):
        return "{}/fi_info ".format(self.fi_info_testpath)

    @property
    def options(self):
        if (self.util_prov):
            opts  = "-f {} -p {};{}".format(self.fabric, self.core_prov, self.util_prov)
        elif (self.core_prov == 'psm3'):
            opts = "-p {}".format(self.core_prov)
        else:
            opts = "-f {} -p {}".format(self.fabric, self.core_prov)

        return opts

    def execute_cmd(self):
        command = self.cmd + self.options
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)


class Fabtest(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, util_prov)
        self.fabtestpath = '{}/bin'.format(self.libfab_installpath)
        self.fabtestconfigpath = '{}/share/fabtests'.format(self.libfab_installpath)

    def get_exclude_file(self):
        path = self.libfab_installpath
        efile_path = '{}/share/fabtests/test_configs'.format(path)

        prov = self.util_prov if self.util_prov else self.core_prov
        efile_old = '{path}/{prov}/{prov}.exclude'.format(path=efile_path,
                      prov=prov)

        if self.util_prov:
            efile = '{path}/{util_prov}/{core_prov}/exclude' \
                    .format(path=efile_path, util_prov=self.util_prov,
                    core_prov=self.core_prov)
        else:
            efile = '{path}/{prov}/exclude'.format(path=efile_path,
                      prov=self.core_prov)

        if os.path.isfile(efile):
            return efile
        elif os.path.isfile(efile_old):
            return efile_old
        else:
            print("Exclude file: {} not found!".format(efile))
            return None

    @property
    def cmd(self):
        return "{}/runfabtests.sh ".format(self.fabtestpath)

    @property
    def options(self):
        opts = "-T 300 -vvv -p {} -S ".format(self.fabtestpath)
        if (self.core_prov != 'shm' and self.nw_interface):
            opts = "{} -s {} ".format(opts, common.get_node_name(self.server,
                    self.nw_interface)) # include common.py
            opts = "{} -c {} ".format(opts, common.get_node_name(self.client,
                    self.nw_interface)) # from common.py

        if (self.core_prov == 'shm'):
            opts = "{} -s {} ".format(opts, self.server)
            opts = "{} -c {} ".format(opts, self.client)
            opts += "-N "

        if (self.ofi_build_mode == 'dl'):
            opts = "{} -t short ".format(opts)
        else:
            opts = "{} -t all ".format(opts)

        if (self.core_prov == 'sockets' and self.ofi_build_mode == 'reg'):
            complex_test_file = '{}/share/fabtests/test_configs/{}/quick.test' \
                                .format(self.libfab_installpath,
                                self.core_prov)
            if (os.path.isfile(complex_test_file)):
                opts = "{} -u {} ".format(opts, complex_test_file)
            else:
                print("{} Complex test file not found".format(self.core_prov))

        if (self.ofi_build_mode != 'reg' or self.core_prov == 'udp'):
            opts = "{} -e \'ubertest,multinode\'".format(opts)

        efile = self.get_exclude_file()
        if efile:
            opts = "{} -R ".format(opts)
            opts = "{} -f {} ".format(opts, efile)

        for key,val in self.env:
            opts = "{options} -E {key}={value} ".format(options = opts,
                    key=key, value=val)

        if self.util_prov:
            opts = "{options} {core};{util} ".format(options=opts,
                    core=self.core_prov, util=self.util_prov)
        else:
            opts = "{options} {core} ".format(options=opts,
                    core=self.core_prov)

        if (self.core_prov == 'shm'):
            opts += "{} {} ".format(self.server, self.server)
        else:
            opts += "{} {} ".format(self.server, self.client)

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
                 hosts, ofi_build_mode, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, util_prov)

        #self.n - number of hosts * number of processes per host
        self.n = 4
        # self.ppn - number of processes per node.
        self.ppn = 2
        self.shmem_dir = '{}/shmem'.format(self.ci_middlewares_path)

    @property
    def cmd(self):
        return "{}/run_shmem.sh ".format(ci_site_config.testpath)

    def options(self, shmem_testname):

        if self.util_prov:
            prov = "{core};{util} ".format(core=self.core_prov,
                    util=self.util_prov)
        else:
            prov = self.core_prov

        opts = "-n {n} ".format(n=self.n)
        opts += "-hosts {server},{client} ".format(server=self.server, \
                                                   client=self.client)
        opts += "-shmem_dir={shmemdir} ".format(shmemdir=self.shmem_dir)
        opts += "-libfabric_path={path}/lib ".format(path=self.libfab_installpath)
        opts += "-prov {provider} ".format(provider=prov)
        opts += "-test {test} ".format(test=shmem_testname)
        opts += "-server {server} ".format(server=self.server)
        opts += "-inf {inf}".format(inf=ci_site_config.interface_map[self.fabric])
        return opts

    @property
    def execute_condn(self):
        #make always true when verbs and sockets are passing
        return True if (self.core_prov == 'tcp') \
                    else False

    def execute_cmd(self, shmem_testname):
        command = self.cmd + self.options(shmem_testname)
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)

class ZeFabtests(Test):
    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, util_prov)
        self.fabtestpath = '{}/bin'.format(self.libfab_installpath)
        self.zefabtest_script_path = '{}'.format(ci_site_config.ze_testpath)
        self.fabtestconfigpath = '{}/share/fabtests'.format(self.libfab_installpath)

    @property
    def cmd(self):
        return '{}/runfabtests_ze.sh '.format(self.zefabtest_script_path)

    @property
    def options(self):
        opts = "-p {} ".format(self.fabtestpath)
        opts += "-B {} ".format(self.fabtestpath)
        opts += "{} {} ".format(self.server, self.client)

        return opts

    @property
    def execute_condn(self):
        #disabled for failures we are investigating
        return False
#        return True if (self.core_prov == 'shm') else False

    def execute_cmd(self):
        curdir = os.getcwd()
        os.chdir(self.fabtestconfigpath)
        command = self.cmd + self.options
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)
        os.chdir(curdir)


class OMPI:
    def __init__(self, core_prov, hosts, libfab_installpath, nw_interface,
                 server, client, environ, ci_middlewares_path, util_prov=None):

        self.ompi_src = '{}/ompi'.format(ci_middlewares_path)
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
            cmd += "export FI_PROVIDER={}; ".format(self.core_prov)
        else:
            cmd += "export FI_PROVIDER={}\\;{}; ".format(self.core_prov,
                                                         self.util_prov)
        cmd += "export I_MPI_FABRICS=ofi; "
        cmd += "export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH; "\
               .format(self.ompi_src)
        cmd += "export LD_LIBRARY_PATH={}/lib/:$LD_LIBRARY_PATH; "\
               .format(self.libfab_installpath)
        cmd += "export PATH={}/bin:$PATH; ".format(self.ompi_src)
        cmd += "export PATH={}/bin:$PATH; ".format(self.libfab_installpath)
        return cmd

    @property
    def options(self):
        opts = "-np {} ".format(self.n)
        hosts = '\',\''.join([':'.join([common.get_node_name(host, \
                         self.nw_interface), str(self.ppn)]) \
                for host in self.hosts])
        opts += "--host \'{}\' ".format(hosts)
        if self.util_prov:
            opts += "--mca mtl_ofi_provider_include {}\\;{} ".format(
                    self.core_prov, self.util_prov)
            opts += "--mca btl_ofi_provider_include {}\\;{} ".format(
                    self.core_prov, self.util_prov)
        else:
            opts += "--mca mtl_ofi_provider_include {} ".format(
                    self.core_prov)
            opts += "--mca btl_ofi_provider_include {} ".format(
                    self.core_prov)
        opts += "--mca orte_base_help_aggregate 0 "
        opts += "--mca mtl ofi "
        opts += "--mca pml cm -tag-output "
        for key, val in self.environ:
            opts = "{} -x {}={} ".format(opts, key, val)

        return opts

    @property
    def cmd(self):
        return "{}/bin/mpirun {}".format(self.ompi_src, self.options)

class MPICH:
    def __init__(self, core_prov, hosts, libfab_installpath, nw_interface,
                 server, client, environ, ci_middlewares_path, util_prov=None):

        self.mpich_src = '{}/mpich'.format(ci_middlewares_path)
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
            cmd += "export FI_PROVIDER={}; ".format(self.core_prov)
        else:
            cmd += "export FI_PROVIDER=\'{};{}\'; ".format(self.core_prov,
                                                           self.util_prov)
        cmd += "export I_MPI_FABRICS=ofi; "
        cmd += "export MPIR_CVAR_CH4_OFI_ENABLE_ATOMICS=0; "
        cmd += "export MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG=1; "
        cmd += "export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH; "\
               .format(self.mpich_src)
        cmd += "export LD_LIBRARY_PATH={}/lib/:$LD_LIBRARY_PATH; "\
               .format(self.libfab_installpath)
        cmd += "export PATH={}/bin:$PATH; ".format(self.mpich_src)
        cmd += "export PATH={}/bin:$PATH; ".format(self.libfab_installpath)
        return cmd

    @property
    def options(self):
        opts = "-n {} ".format(self.n)
        opts += "-ppn {} ".format(self.ppn)
        opts += "-hosts {},{} ".format(common.get_node_name(self.server,
                                       self.nw_interface),
                                       common.get_node_name(self.client,
                                       self.nw_interface))
        for key, val in self.environ:
            opts = "{} -genv {} {} ".format(opts, key, val)

        return opts

    @property
    def cmd(self):
        return "{}/bin/mpirun {}".format(self.mpich_src, self.options)


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
        cmd = "bash -c \'source {}/env/vars.sh -i_mpi_ofi_internal=0; "\
              .format(self.impi_src)
        if (self.util_prov):
            cmd += "export FI_PROVIDER={}; ".format(self.core_prov)
        else:
            cmd += "export FI_PROVIDER=\'{};{}\'; ".format(self.core_prov,
                                                           self.util_prov)
        cmd += "export I_MPI_FABRICS=ofi; "
        cmd += "export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH; "\
               .format(self.impi_src)
        cmd += "export LD_LIBRARY_PATH={}/lib/release:$LD_LIBRARY_PATH; "\
               .format(self.impi_src)
        cmd += "export LD_LIBRARY_PATH={}/lib/:$LD_LIBRARY_PATH; "\
               .format(self.libfab_installpath)
        cmd += "export PATH={}/bin:$PATH; ".format(self.libfab_installpath)
        return cmd

    @property
    def options(self):
        opts = "-n {} ".format(self.n)
        opts += "-ppn {} ".format(self.ppn)
        opts += "-hosts {},{} ".format(common.get_node_name(self.server,
                                       self.nw_interface),
                                       common.get_node_name(self.client,
                                       self.nw_interface))
        for key, val in self.environ:
            opts = "{} -genv {} {} ".format(opts, key, val)

        return opts

    @property
    def cmd(self):
        return "{}/bin/mpiexec {}".format(self.impi_src, self.options)


class IMBtests(Test):
    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, mpitype, ofi_build_mode, test_group, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, util_prov)

        self.test_name = testname
        self.core_prov = core_prov
        self.util_prov = util_prov
        self.test_group = test_group
        self.mpi_type = mpitype
        self.mpi = ''
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
            self.mpi = IMPI(self.core_prov, self.hosts,
                            self.libfab_installpath, self.nw_interface,
                            self.server, self.client, self.env, self.util_prov)
            self.imb_src = ci_site_config.impi_root
        elif (self.mpi_type == 'ompi'):
            self.mpi = OMPI(self.core_prov, self.hosts,
                             self.libfab_installpath, self.nw_interface,
                             self.server, self.client, self.env,
                             self.ci_middlewares_path, self.util_prov)
            self.imb_src = '{}/ompi/imb'.format(self.ci_middlewares_path)
        elif (self.mpi_type == 'mpich'):
            self.mpi = MPICH(self.core_prov, self.hosts,
                             self.libfab_installpath, self.nw_interface,
                             self.server, self.client, self.env,
                             self.ci_middlewares_path, self.util_prov)
            self.imb_src = '{}/mpich/imb'.format(self.ci_middlewares_path)

    @property
    def execute_condn(self):
        # Mpich and ompi are excluded to save time. Run manually if needed
        return True if (self.mpi_type == 'impi') else False

    def imb_cmd(self, imb_test):
        print("Running IMB-{}".format(imb_test))
        cmd = "{}/bin/IMB-{} ".format(self.imb_src, imb_test)
        if (self.test_name != 'MT'):
            cmd += "-iter {} ".format(self.iter)

        if (len(self.include[imb_test]) > 0):
            cmd += "-include {} ".format(','.join(self.include[imb_test]))

        if (len(self.exclude[imb_test]) > 0):
            cmd += "-exclude {} ".format(','.join(self.exclude[imb_test]))

        return cmd

    def execute_cmd(self):
        for test_type in self.imb_tests[self.test_group]:
                outputcmd = shlex.split(self.mpi.env + self.mpi.cmd + \
                                        self.imb_cmd(test_type) + '\'')
                common.run_command(outputcmd)


class OSUtests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, mpitype, ofi_build_mode, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, util_prov)

        self.n_ppn = {
                          'pt2pt':      (2, 1),
                          'collective': (4, 2),
                          'one-sided':  (2, 1),
                          'startup':    (2, 1)
                     }
        self.osu_src = '{}/{}/osu/libexec/osu-micro-benchmarks/mpi/'. \
                            format(self.ci_middlewares_path, mpitype)
        self.mpi_type = mpitype
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

    @property
    def execute_condn(self):
        # mpich-tcp and ompi-tcp are the only osu test combinations failing
        return False if ((self.mpi == 'mpich' and self.core_prov == 'tcp') or \
                          self.mpi == 'ompi') \
                    else True

    def osu_cmd(self, test_type, test):
        print("Running OSU-{}-{}".format(test_type, test))
        cmd = '{}/{}/{} '.format(self.osu_src, test_type, test)
        return cmd

    def execute_cmd(self):
        assert(self.osu_src)
        p = re.compile('osu_put*')
        for root, dirs, tests in os.walk(self.osu_src):
            for test in tests:
                self.mpi.n = self.n_ppn[os.path.basename(root)][0]
                self.mpi.ppn = self.n_ppn[os.path.basename(root)][1]
        
                if (test == 'osu_latency_mp' and self.core_prov == 'verbs'):
                    self.env.append(('IBV_FORK_SAFE', '1'))

                if(p.search(test) == None):
                    osu_command = self.osu_cmd(os.path.basename(root), test)
                    outputcmd = shlex.split(self.mpi.env + self.mpi.cmd + \
                                            osu_command + '\'')
                    common.run_command(outputcmd)

                if (test == 'osu_latency_mp' and self.core_prov == 'verbs'):
                    self.env.remove(('IBV_FORK_SAFE', '1'))


class MpichTestSuite(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, mpitype, ofi_build_mode, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, util_prov)

        self.mpichsuitepath = '{}/{}/mpichsuite/test/mpi/' \
                              .format(self.ci_middlewares_path, mpitype)
        self.pwd = os.getcwd()
        self.mpi_type = mpitype
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

    def testgroup(self, testgroupname):
        testpath = '{}/{}'.format(self.mpichsuitepath, testgroupname)
        tests = []
        print(f'{testpath}/testlist')
        with open('{}/testlist'.format(testpath)) as file:
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
        return True if (self.mpi_type == 'impi' or \
                       (self.mpi_type == 'mpich' and \
                        self.core_prov == 'verbs')) \
                    else False

    def execute_cmd(self, testgroupname):
        print("Running Tests: " + testgroupname)
        tests = []
        time = None
        os.chdir("{}/{}".format(self.mpichsuitepath, testgroupname))
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
            testcmd = "./{}".format(testname)
            outputcmd = shlex.split(self.mpi.env + self.mpi.cmd + testcmd + '\'')
            common.run_command(outputcmd)
        os.chdir(self.pwd)


class OneCCLTests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, util_prov)

        self.n = 2
        self.ppn = 1
        self.oneccl_path = '{}/oneccl/build'.format(self.ci_middlewares_path)

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
        return "{}/run_oneccl.sh ".format(ci_site_config.testpath)

    def options(self, oneccl_test):
        opts = "-n {n} ".format(n=self.n)
        opts += "-ppn {ppn} ".format(ppn=self.ppn)
        opts += "-hosts {server},{client} ".format(server=self.server,
                                                   client=self.client)
        opts += "-prov '{provider}' ".format(provider=self.core_prov)
        opts += "-test {test_suite} ".format(test_suite=oneccl_test)
        opts += "-libfabric_path={path}/lib " \
                .format(path=self.libfab_installpath)
        opts += '-oneccl_root={oneccl_path}' \
                .format(oneccl_path=self.oneccl_path)
        return opts

    @property
    def execute_condn(self):
        return True


    def execute_cmd(self, oneccl_test):
        if oneccl_test == 'examples':
                for test in self.examples_tests:
                        command = self.cmd + self.options(oneccl_test) + \
                                  " {}".format(test)
                        outputcmd = shlex.split(command)
                        common.run_command(outputcmd)
        elif oneccl_test == 'functional':
                for test in self.functional_tests:
                        command = self.cmd + self.options(oneccl_test) + \
                                  " {}".format(test)
                        outputcmd = shlex.split(command)
                        common.run_command(outputcmd)

class OneCCLTestsGPU(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 hosts, ofi_build_mode, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         hosts, ofi_build_mode, util_prov)

        self.n = 2
        self.ppn = 4
        self.oneccl_path = '{}/oneccl_gpu/build'.format(self.ci_middlewares_path)

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
        return "{}/run_oneccl_gpu.sh ".format(ci_site_config.testpath)

    def options(self, oneccl_test_gpu):
        opts = "-n {n} ".format(n=self.n)
        opts += "-ppn {ppn} ".format(ppn=self.ppn)
        opts += "-hosts {server},{client} ".format(server=self.server,
                                                   client=self.client)
        opts += "-test {test_suite} ".format(test_suite=oneccl_test_gpu)
        opts += "-libfabric_path={path}/lib " \
                .format(path=self.libfab_installpath)
        opts += '-oneccl_root={oneccl_path}' \
                .format(oneccl_path=self.oneccl_path)
        return opts

    @property
    def execute_condn(self):
        return True


    def execute_cmd(self, oneccl_test_gpu):
        if oneccl_test_gpu == 'examples':
                for test in self.examples_tests:
                        command = self.cmd + self.options(oneccl_test_gpu) + \
                                  " {}".format(test)
                        outputcmd = shlex.split(command)
                        common.run_command(outputcmd)
        elif oneccl_test_gpu == 'functional':
                for test in self.functional_tests:
                        command = self.cmd + self.options(oneccl_test_gpu) + \
                                  " {}".format(test)
                        outputcmd = shlex.split(command)
                        common.run_command(outputcmd)
