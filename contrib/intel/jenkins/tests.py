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
        if (self.core_prov == 'verbs' and self.nw_interface):
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
            opts = "{} -e \'multinode,ubertest\' ".format(opts)

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
        return True if (self.core_prov == 'shm') else False

    def execute_cmd(self):
        curdir = os.getcwd()
        os.chdir(self.fabtestconfigpath)
        command = self.cmd + self.options
        outputcmd = shlex.split(command)
        common.run_command(outputcmd)
        os.chdir(curdir)

class MpiTests(Test):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 mpitype, hosts, ofi_build_mode, imb_group, util_prov=None):

        super().__init__(jobname, buildno, testname, core_prov,
                         fabric, hosts, ofi_build_mode, util_prov)
        self.mpi = mpitype
        self.imb_group=imb_group

    @property
    def cmd(self):
        if (self.mpi == 'impi' or self.mpi == 'mpich'):
            self.testpath = ci_site_config.testpath
            return "{}/run_{}.sh ".format(self.testpath,self.mpi)
        elif(self.mpi == 'ompi'):
            self.testpath = "{}/ompi/bin".format(self.ci_middlewares_path)
            return "{}/mpirun ".format(self.testpath)

    @property
    def options(self):
        opts = []
        if (self.mpi == 'impi' or self.mpi == 'mpich'):
            opts = "-n {} ".format(self.n)
            opts += "-ppn {} ".format(self.ppn)
            opts += "-hosts {},{} ".format(self.server,self.client)

            if (self.mpi == 'impi'):
                opts = "{} -mpi_root={} ".format(opts,
                        ci_site_config.impi_root)
            else:
                opts = "{} -mpi_root={}/mpich".format(opts,
                        self.ci_middlewares_path)

            opts = "{} -libfabric_path={}/lib ".format(opts,
                    self.libfab_installpath)

            if self.util_prov:
                opts = "{options} -prov {core};{util} ".format(options=opts,
                        core=self.core_prov, util=self.util_prov)
            else:
                opts = "{} -prov {} ".format(opts, self.core_prov)

            for key, val in self.env:
                opts = "{} -genv {} {} ".format(opts, key, val)

        elif (self.mpi == 'ompi'):
            opts = "-np {} ".format(self.n)
            hosts = ','.join([':'.join([host,str(self.ppn)]) \
                    for host in self.hosts])

            opts = "{} --host {} ".format(opts, hosts)

            if self.util_prov:
                opts = "{} --mca mtl_ofi_provider_include {};{} ".format(opts,
                        self.core_prov,self.util_prov)
            else:
                opts = "{} --mca mtl_ofi_provider_include {} ".format(opts,
                        self.core_prov)

            opts += "--mca orte_base_help_aggregate 0 "
            opts += "--mca mtl ofi --mca pml cm -tag-output "
            for key,val in self.env:
                opts = "{} -x {}={} ".format(opts,key,val)
        return opts

    @property
    def mpi_gen_execute_condn(self):
        return True if (self.mpi == 'impi' or self.mpi == 'mpich') \
                    else False

class IMBtests:
    def __init__(self, test_name, core_prov, util_prov):
        self.test_name = test_name
        self.core_prov = core_prov
        self.util_prov = util_prov
        # Iters are limited for time constraints
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

    @property
    def imb_cmd(self):
        print("Running IMB-{}".format(self.test_name))
        cmd = "{}/bin/IMB-{} ".format(ci_site_config.impi_root, self.test_name)
        if (self.test_name != 'MT'):
            cmd += "-iter {} ".format(self.iter)

        if (len(self.include[self.test_name]) > 0):
            cmd += "-include {} ".format(','.join(self.include[self.test_name]))

        if (len(self.exclude[self.test_name]) > 0):
            cmd += "-exclude {} ".format(','.join(self.exclude[self.test_name]))
        return cmd

    @property
    def execute_condn(self):
        return True

class MpiTestIMB(MpiTests):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 mpitype, hosts, ofi_build_mode, test_group, util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         mpitype, hosts, ofi_build_mode, util_prov)

        self.test_group = test_group
        self.n = 4
        self.ppn = 1
        self.imb_tests = {
                             '1':[
                                     'MPI1',
                                     'P2P'
                                 ],
                             '2':[
                                     'EXT',
                                     'IO'
                                 ],
                             '3':[
                                     'NBC',
                                     'RMA',
                                     'MT'
                                 ]
                         }

    @property
    def execute_condn(self):
        return True if (self.mpi == 'impi') else False

    def execute_cmd(self):
        command = self.cmd + self.options
        for test_type in self.imb_tests[self.test_group]:
            self.test_obj = IMBtests(test_type, self.core_prov, self.util_prov)
            if (self.test_obj.execute_condn):
                outputcmd = shlex.split(command + self.test_obj.imb_cmd)
                common.run_command(outputcmd)
            else:
                print("IMB-{} not run".format(test_type))


class MpichTestSuite(MpiTests):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
		 mpitype, hosts, ofi_build_mode, imb_group=None,
                 util_prov=None):
            super().__init__(jobname, buildno, testname, core_prov, fabric,
			     mpitype,  hosts, ofi_build_mode, util_prov)
            self.mpichsuitepath = '{}/{}/mpichsuite/test/mpi/' \
                                  .format(self.ci_middlewares_path, self.mpi)
            self.pwd = os.getcwd()

    def testgroup(self, testgroupname):

        testpath = '{}/{}'.format(self.mpichsuitepath, testgroupname)
        tests = []
        with open('{}/testlist'.format(testpath)) as file:
            for line in file:
                if(line[0] != '#' and  line[0] != '\n'):
                    tests.append((line.rstrip('\n')).split(' '))

        return tests

    def options(self, nprocs, timeout=None):
        if (self.mpi == 'impi' or self.mpi == 'mpich'):
            if (self.mpi == 'impi'):
                mpiroot = ci_site_config.impi_root
            else:
                mpiroot = '{}/mpich'.format(self.ci_middlewares_path)
            if (self.util_prov):
                prov = '\"{};{}\"'.format(self.core_prov, self.util_prov)
            else:
                prov = self.core_prov

            if (timeout != None):
                os.environ['MPIEXEC_TIMEOUT']=timeout

            opts = "-n {np} ".format(np=nprocs)
            opts += "-hosts {s},{c} ".format(s=self.server, c=self.client)
            opts += "-mpi_root={mpiroot} ".format(mpiroot=mpiroot)
            opts += "-libfabric_path={installpath}/lib " \
                    .format(installpath=self.libfab_installpath)
            opts += "-prov {provider} ".format(provider=prov)

        elif (self.mpi == 'ompi'):
            print(self.mpi)

        return opts

    @property
    def execute_condn(self):
        # MPICH tcp mpich testsuite hangs
        # MPICH sockets mpich testsuite hangs
        return True if (self.mpi == 'impi' or \
                       (self.mpi == 'mpich' and \
                        self.core_prov == 'verbs')) \
                    else False

    def execute_cmd(self, testgroupname):
        print("Running Tests: " + testgroupname)
        tests = []
        time = None
        os.chdir("{}/{}".format(self.mpichsuitepath,testgroupname))
        tests = self.testgroup(testgroupname)
        for test in tests:
            testname = test[0]
            nprocs = test[1]
            args = test[2:]
            for item in args:
               itemlist =  item.split('=')
               if (itemlist[0] == 'timelimit'):
                   time = itemlist[1]
            opts = self.options(nprocs, timeout=time)
            testcmd = self.cmd + opts +"./{}".format(testname)
            outputcmd = shlex.split(testcmd)
            common.run_command(outputcmd)
        os.chdir(self.pwd)


class MpiTestOSU(MpiTests):

    def __init__(self, jobname, buildno, testname, core_prov, fabric,
                 mpitype, hosts, ofi_build_mode, imb_group=None,
                 util_prov=None):
        super().__init__(jobname, buildno, testname, core_prov, fabric,
                         mpitype, hosts, ofi_build_mode, util_prov)

        self.n = 4
        self.ppn = 2
        self.two_proc_tests = {
                                  'osu_latency',
                                  'osu_bibw',
                                  'osu_latency_mt',
                                  'osu_bw',
                                  'osu_get_latency',
                                  'osu_fop_latency',
                                  'osu_acc_latency',
                                  'osu_get_bw',
                                  'osu_put_latency',
                                  'osu_put_bw',
                                  'osu_put_bibw',
                                  'osu_cas_latency',
                                  'osu_get_acc_latency',
                                  'osu_latency_mp'
                              }
       #these tests have race conditions or segmentation faults
       #self.disable = {
       #                   'osu_allgather',
       #                   'osu_allgatherv',
       #                   'osu_allreduce',
       #                   'osu_alltoall'
       #                   'osu_alltoallv',
       #                   'osu_iallgather',
       #                   'osu_iallgatherv',
       #                   'osu_ialltoall',
       #                   'osu_ialltoallv',
       #                   'osu_ialltoallw',
       #                   'osu_ibarrier'
       #               }

        self.osu_mpi_path = '{}/{}/osu/libexec/osu-micro-benchmarks/mpi/'. \
                            format(self.ci_middlewares_path, mpitype)

    @property
    def execute_condn(self):
        # see disable list for ompi failures
        return True if ((self.mpi == 'impi' or self.mpi == 'mpich') and \
                        (self.core_prov == 'verbs')) \
                    else False

    def execute_cmd(self):
        assert(self.osu_mpi_path)
        p = re.compile('osu_put*')
        for root, dirs, tests in os.walk(self.osu_mpi_path):
            for test in tests:
#                if test in self.disable:
#                    continue

                if test in self.two_proc_tests:
                    self.n=2
                    self.ppn=1
                else:
                    self.n=4
                    self.ppn=2

                if (test == 'osu_latency_mp' and self.core_prov == 'verbs'):
                    self.env.append(('IBV_FORK_SAFE', '1'))

                if(p.search(test) == None):
                    launcher = self.cmd + self.options
                    osu_cmd = os.path.join(root, test)
                    command = launcher + osu_cmd
                    outputcmd = shlex.split(command)
                    common.run_command(outputcmd)

                if (test == 'osu_latency_mp' and self.core_prov == 'verbs'):
                    self.env.remove(('IBV_FORK_SAFE', '1'))

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

