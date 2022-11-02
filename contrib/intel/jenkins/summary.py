from abc import ABC, abstractmethod
import os
from pickle import FALSE
import sys

# add jenkins config location to PATH
sys.path.append(os.environ['CI_SITE_CONFIG'])

import ci_site_config
import argparse
import common

verbose = False
spacing='\t'

class Summarizer(ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "print_results")
            and callable(subclass.print_results)
            and hasattr(subclass, "check_name")
            and callable(subclass.check_name)
            and hasattr(subclass, "check_pass")
            and callable(subclass.check_pass)
            and hasattr(subclass, "check_fail")
            and callable(subclass.check_fail)
            and hasattr(subclass, "check_exclude")
            and callable(subclass.check_exclude)
            and hasattr(subclass, "read_file")
            and callable(subclass.read_file)
            and hasattr(subclass, "run")
            and callable(subclass.run)
            or NotImplemented
        )

    @abstractmethod
    def __init__(self, log_dir, prov, build_mode, file_name, stage_name):
        self.log_dir = log_dir
        self.prov = prov
        self.mode = build_mode
        self.file_name = file_name
        self.stage_name = stage_name
        self.file_path = os.path.join(self.log_dir, self.file_name)
        self.exists = os.path.exists(self.file_path)
        self.log = None
        self.line = ''
        self.result = ''
        self.passes = 0
        self.fails = 0
        self.failed_tests = []
        self.excludes = 0
        self.excluded_tests = []
        self.test_name_string='no_test'

    def print_results(self):
        total = self.passes + self.fails
        # log was empty or not valid
        if not total:
            return

        percent = self.passes/total * 100
        print(f"{spacing}{self.stage_name}: ".ljust(40), end='')
        print(f"{self.passes}/{total} ".ljust(10), end='')
        print(f"= {percent:.2f}%".ljust(12), end = '')
        print("Pass")
        if self.fails:
            print(f"{spacing}\tFailed tests: {self.fails}")
            for test in self.failed_tests:
                    print(f'{spacing}\t\t{test}')
        if (verbose):
            if self.excludes:
                print(f"{spacing}\tExcluded/Notrun tests: {self.excludes} ")
                for test in self.excluded_tests:
                    print(f'{spacing}\t\t{test}')

    def check_name(self):
        return
 
    def check_pass(self):
        return

    def check_fail(self):
        if "exiting with" in self.line:
            self.fails += 1

    def check_exclude(self):
        return

    def check_line(self):
        self.check_name()
        self.check_pass()
        self.check_fail()
        self.check_exclude()

    def read_file(self):
        self.log = open(self.file_path, 'r')
        self.line = self.log.readline().lower()
        while self.line:
            self.check_line()
            self.line = self.log.readline().lower()

        self.log.close()

    def summarize(self):
        if not self.exists:
            return 0

        self.read_file()
        self.print_results()
        return int(self.fails)

class FiInfoSummarizer(Summarizer):
    def __init__(self, log_dir, prov, build_mode):
        super().__init__(log_dir, prov, build_mode,
                         f'{prov}_fi_info_{build_mode}',
                         f"{prov} fi_info {build_mode}")

    def check_fail(self):
        if "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(f"fi_info {self.prov}")

    def read_file(self):
        self.log = open(self.file_name, 'r')
        line = self.log.readline()
        while line:
            self.check_line()
            line = self.log.readline()

        if not self.fails:
            self.passes += 1

        self.log.close()

class FabtestsSummarizer(Summarizer):
    def __init__(self, log_dir, prov, build_mode):
        super().__init__(log_dir, prov, build_mode,
                         f'{prov}_fabtests_{mode}',
                         f"{prov} fabtests {mode}")
        self.result = ''
        self.result_line = ''

    def check_name(self):
        # don't double count ubertest output
        if 'ubertest' in self.line and 'client_cmd:' in self.line:
            while 'name:' not in self.line: # skip past client output in ubertest
                self.line = self.log.readline().lower()

        if 'name:' in self.line:
            test_name = self.line.split()[2:]
            self.test_name_string = ' '.join(test_name)

    def get_result_line(self):
        if 'result:' in self.line:
            self.result_line = self.line.split()
            # lines can look like 'result: Pass' or
            # 'Ending test 1 result: Success'
            self.result = (self.result_line[self.result_line.index(
                           'result:') + 1]).lower()

    def check_pass(self):
        self.get_result_line()
        if self.result == 'pass' or self.result == 'success':
            self.passes += 1

        self.result = ''

    def check_fail(self):
        self.get_result_line()
        if self.result == 'fail':
            self.fails += 1
            if 'ubertest' in self.test_name_string:
                idx = (self.result_line.index('result:') - 1)
                ubertest_number = int((self.result_line[idx].split(',')[0]))
                self.failed_tests.append(f"{self.test_name_string}: " \
                                    f"{ubertest_number}")
            else:
                self.failed_tests.append(self.test_name_string)

        if "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(self.test_name_string)

        self.result = ''

    def check_exclude(self):
        self.get_result_line()
        if self.result == 'excluded' or self.result == 'notrun':
            self.excludes += 1
            self.excluded_tests.append(self.test_name_string)

        self.result = ''

class ZeSummarizer(Summarizer):
    def __init__(self, log_dir, prov, test_type, build_mode):
        self.test_type = test_type
        self.result = ''
        self.result_line = ''
        super().__init__(log_dir, prov, build_mode,
                         f'ze-{prov}_{self.test_type}_{build_mode}',
                         f"ze {prov} {test_type} {build_mode}")

    def check_name(self):
        # don't double count ubertest output
        if 'ubertest' in self.line and 'client_cmd:' in self.line:
            while 'name:' not in self.line: # skip past client output in ubertest
                self.line = self.log.readline().lower()

        if 'name:' in self.line:
            test_name = self.line.split()[2:]
            self.test_name_string = ' '.join(test_name)

    def get_result_line(self):
        if 'result:' in self.line:
            self.result_line = self.line.split()
            # lines can look like 'result: Pass' or
            # 'Ending test 1 result: Success'
            self.result = (self.result_line[self.result_line.index(
                           'result:') + 1]).lower()

    def check_pass(self):
        self.get_result_line()
        if self.result == 'pass' or self.result == 'success':
            self.passes += 1

        self.result = ''

    def check_fail(self):
        self.get_result_line()
        if self.result == 'fail':
            fails += 1
            if 'ubertest' in self.test_name_string:
                idx = (self.result_line.index('result:') - 1)
                ubertest_number = int((self.result_line[idx].split(',')[0]))
                self.failed_tests.append(f"{self.test_name_string}: " \
                                    f"{ubertest_number}")
            else:
                self.failed_tests.append(self.test_name_string)

        if "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(self.test_name_string)

        self.result = ''

        return

    def check_exclude(self):
        self.get_result_line()
        if self.result == 'excluded' or self.result == 'notrun':
            self.excludes += 1
            self.excluded_tests.append(self.test_name_string)
        self.result = ''

class OnecclSummarizer(Summarizer):
    def __init__(self, log_dir, prov, build_mode):
        super().__init__(log_dir, prov, build_mode,
                         f'{prov}_onecclgpu_{mode}' if 'GPU' in prov else \
                         f'{prov}_oneccl_{mode}',
                         f"{prov} {mode}")
        self.file_path = os.path.join(self.log_dir, self.file_name)
        self.exists = os.path.exists(self.file_path)
        self.name = 'no_test'

    def check_name(self):
        #lines look like path/run_oneccl.sh ..... -test examples ..... test_name
        if " -test" in self.line:
            tokens = self.line.split()
            self.name = f"{tokens[tokens.index('-test') + 1]} " \
                   f"{tokens[len(tokens) - 1]}"

    def check_pass(self):
        self.passes += 1 if 'passed' in self.line else 0

    def check_fail(self):
        if 'failed' in self.line or "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(self.name)

class ShmemSummarizer(Summarizer):
    def __init__(self, log_dir, prov, build_mode):
        super().__init__(log_dir, prov, build_mode,
                         f'SHMEM_{prov}_shmem_{build_mode}',
                         f"shmem {prov} {build_mode}")
        if self.prov == 'uh':
            self.keyphrase = 'summary'
            # Failed
        if self.prov == 'isx':
            self.keyphrase = 'scaling'
            # Failed
        if self.prov == 'prk':
            self.keyphrase = 'solution'
            # ERROR:
        self.name = 'no_test'

    def check_uh(self):
        # (test_002) Running test_shmem_atomics.x: Test all atomics... OK
        # (test_003) Running test_shmem_barrier.x: Tests barrier ... Failed
        if "running test_" in self.line:
            tokens = self.line.split()
            for token in tokens:
                if 'test_' in token:
                    self.name = token
            if tokens[len(tokens) - 1] == 'ok':
                self.passes += 1
            else:
                self.fails += 1
                self.failed_tests.append(self.name)
        # Summary
        # x/z Passed.
        # y/z Failed.
        if self.keyphrase in self.line: #double check
            passed = self.log.readline().lower()
            failed = self.log.readline().lower()
            if self.passes != int(passed.split()[1].split('/')[0]):
                print(f"passes {self.passes} do not match log reported passes " \
                        f"{int(passed.split()[1].split('/')[0])}")
            if self.fails != int(failed.split()[1].split('/')[0]):
                print(f"fails {self.fails} does not match log fails " \
                        f"{int(failed.split()[1].split('/')[0])}")

    def check_prk(self):
        if self.keyphrase in self.line:
            self.passes += 1
        if 'error:' in self.line or "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(f"{self.prov} {self.passes + self.fails}")
        if 'test(s)' in self.line:
            if int(self.line.split()[0]) != self.fails:
                print(f"fails {self.fails} does not match log reported fails " \
                    f"{int(self.line.split()[0])}")

    def check_isx(self):
        if self.keyphrase in self.line:
            self.passes += 1
        if ('failed' in self.line and 'test(s)' not in self.line) or \
            "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(f"{self.prov} {self.passes + self.fails}")
        if 'test(s)' in self.line:
            if int(self.line.split()[0]) != self.fails:
                print(f"fails {self.fails} does not match log reported fails " \
                        f"{int(self.line.split()[0])}")

    def check_fails(self):
        if "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(f"{self.prov} {self.passes + self.fails}")

    def check_line(self):
        if self.prov == 'uh':
            self.check_uh()
        if self.prov == 'isx':
            self.check_isx()
        if self.prov == 'prk':
            self.check_prk()
        self.check_fails()

class MpichTestSuiteSummarizer(Summarizer):
    def __init__(self, log_dir, prov, mpi, build_mode):
        super().__init__(log_dir, prov, build_mode,
                         f'MPICH testsuite_{prov}_{mpi}_mpichtestsuite_{build_mode}',
                         f"{prov} {mpi} mpichtestsuite {build_mode}")

        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = 'mpiexec'
        else:
            self.run = 'mpirun'

    def check_name(self):
        if self.run in self.line:
            self.name = self.line.split()[len(self.line.split()) - 1].split('/')[1]
            #assume pass
            self.passes += 1

    def check_fail(self):
        # Fail cases take away assumed pass
        if "exiting with" in self.line:
            self.fails += 1
            self.passes -= 1
            self.failed_tests.append(f'{self.name}')
            #skip to next test
            while self.run not in self.line:
                self.line = self.log.readline().lower()

class ImbSummarizer(Summarizer):
    def __init__(self, log_dir, prov, mpi, build_mode):
        super().__init__(log_dir, prov, build_mode,
                         f'MPI_{prov}_{mpi}_IMB_{build_mode}',
                         f"{prov} {mpi} IMB {build_mode}")

        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = 'mpiexec'
        else:
            self.run = 'mpirun'
        self.test_type = ''

    def check_type(self):
        if 'part' in self.line:
            self.test_type = self.line.split()[len(self.line.split()) - 2]

    def check_name(self):
        if "benchmarking" in self.line:
            self.name = self.line.split()[len(self.line.split()) - 1]

    def check_pass(self):
        if "benchmarking" in self.line:
            self.passes += 1

    def check_fail(self):
        if "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(f"{self.test_type} {self.name}")
            self.passes -= 1

    def check_line(self):
        self.check_type()
        self.check_name()
        self.check_pass()
        self.check_fail()
        super().check_exclude()

class OsuSummarizer(Summarizer):
    def __init__(self, log_dir, prov, mpi, build_mode):
        super().__init__(log_dir, prov, build_mode,
                         f'MPI_{prov}_{mpi}_osu_{build_mode}',
                         f"{prov} {mpi} OSU {build_mode}")
        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = 'mpiexec'
        else:
            self.run = 'mpirun'

        self.type = ''
        self.tokens = []

    def get_tokens(self):
        if "# osu" in self.line:
            self.tokens = self.line.split()
        else:
            self.tokens = []

    def check_name(self):
        if 'osu' in self.tokens:
            self.name = " ".join(self.tokens[self.tokens.index('osu') + \
                        1:self.tokens.index('test')])

    def check_type(self):
        if self.tokens:
            self.test_type = self.tokens[1]

    def check_pass(self):
        if 'osu' in self.tokens:
            # Assume pass
            self.passes += 1

    def check_fail(self):
        if "exiting with" in self.line:
            self.fails += 1
            self.failed_tests.append(f"{self.test_type} {self.name}")
            # Remove assumed pass
            self.passes -= 1

    def check_line(self):
        self.get_tokens()
        self.check_name()
        self.check_type()
        self.check_pass()
        self.check_fail()
        super().check_exclude()

if __name__ == "__main__":
#read Jenkins environment variables
    # In Jenkins,  JOB_NAME  = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
    # job name is better to use to distinguish between builds of different
    # jobs but with same branch name.
    jobname = os.environ['JOB_NAME']
    buildno = os.environ['BUILD_NUMBER']

    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_item', help="functional test to summarize",
                         choices=['fabtests', 'imb', 'osu', 'mpichtestsuite',
                         'oneccl', 'shmem', 'ze', 'all'])
    parser.add_argument('--ofi_build_mode', help="select buildmode debug or dl",
                        choices=['dbg', 'dl', 'reg'], default='all')
    parser.add_argument('-v', help="Verbose mode. Print excluded tests", \
                        action='store_true')

    args = parser.parse_args()
    verbose = args.v

    args = parser.parse_args()
    summary_item = args.summary_item

    mpi_list = ['impi', 'mpich', 'ompi']

    if (args.ofi_build_mode):
        ofi_build_mode = args.ofi_build_mode
    else:
        ofi_build_mode = 'reg'

    log_dir = f'{ci_site_config.install_dir}/{jobname}/{buildno}/log_dir'
    ret = 0
    err = 0

    build_modes = ['reg', 'dbg', 'dl']
    for mode in build_modes:
        if ofi_build_mode != 'all' and mode != ofi_build_mode:
            continue

        print(f"Summarizing {mode} build mode:")
        if summary_item == 'fabtests' or summary_item == 'all':
            for prov,util in common.prov_list:
                if util:
                    prov = f'{prov}-{util}'

                ret = FabtestsSummarizer(log_dir, prov, mode).summarize()
                err += ret if ret else 0
                ret = FiInfoSummarizer(log_dir, prov, mode).summarize()
                err += ret if ret else 0

        if summary_item == 'imb' or summary_item == 'all':
            for mpi in mpi_list:
                for item in ['tcp-rxm', 'verbs-rxm', 'net']:
                    ret = ImbSummarizer(log_dir, item, mpi, mode).summarize()
                    err += ret if ret else 0

        if summary_item == 'osu' or summary_item == 'all':
            for mpi in mpi_list:
                    for item in ['tcp-rxm', 'verbs-rxm']:
                        ret = OsuSummarizer(log_dir, item, mpi,
                                            mode).summarize()
                        err += ret if ret else 0

        if summary_item == 'mpichtestsuite' or summary_item == 'all':
            for mpi in mpi_list:
                    for item in ['tcp-rxm', 'verbs-rxm', 'sockets']:
                        ret = MpichTestSuiteSummarizer(log_dir, item,
                                                       mpi, mode).summarize()
                        err += ret if ret else 0

        if summary_item == 'oneccl' or summary_item == 'all':
            ret = OnecclSummarizer(log_dir, 'oneCCL', mode).summarize()
            err += ret if ret else 0
            ret = OnecclSummarizer(log_dir, 'oneCCL-GPU', mode).summarize()
            err += ret if ret else 0

        if summary_item == 'shmem' or summary_item == 'all':
            ret = ShmemSummarizer(log_dir, 'uh', mode).summarize()
            err += ret if ret else 0
            ret = ShmemSummarizer(log_dir, 'prk', mode).summarize()
            err += ret if ret else 0
            ret = ShmemSummarizer(log_dir, 'isx', mode).summarize()
            err += ret if ret else 0

        if summary_item == 'ze' or summary_item == 'all':
            test_types = ['h2d', 'd2d', 'xd2d']
            for type in test_types:
                ret = ZeSummarizer(log_dir, 'shm', type, mode).summarize()
                err += ret if ret else 0

    exit(err)
