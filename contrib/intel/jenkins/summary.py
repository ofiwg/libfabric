from abc import ABC, abstractmethod
import shutil
from tempfile import NamedTemporaryFile
from datetime import datetime
from typing import Tuple
import os
from pickle import FALSE
import sys

# add jenkins config location to PATH
sys.path.append(os.environ['CLOUDBEES_CONFIG'])

import cloudbees_config
import argparse
import common

verbose = False

class Release:
    def __init__(self, log_dir, output_file, logger, release_num):
        self.log_dir = log_dir
        self.output_file = output_file
        self.logger = logger
        self.release_num = release_num

    def __log_entire_file(self, file_name):
        with open(file_name) as f:
            for line in f:
                self.logger.log(line, end_delimiter = '')

    def __append_release_changes(self, file_name):
        if os.path.exists(file_name):
            self.__log_entire_file(file_name)

    def add_release_changes(self):
        self.logger.log(F"Release number: {self.release_num}")
        self.__append_release_changes(f'{self.log_dir}/Makefile.am.diff')
        self.__append_release_changes(f'{self.log_dir}/configure.ac.diff')

class Logger:
    def __init__(self, output_file, release):
        self.output_file = output_file
        self.release = release
        self.padding = '\t'

    def log(self, line, end_delimiter='\n', lpad=0, ljust=0):
        print(f'{self.padding * lpad}{line}'.ljust(ljust), end = end_delimiter)
        if (self.release):
            self.output_file.write(
                f'{self.padding * lpad}{line}{end_delimiter}'
            )

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
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        self.logger = logger
        self.log_dir = log_dir
        self.prov = prov
        self.file_name = file_name
        self.stage_name = stage_name
        self.file_path = os.path.join(self.log_dir, self.file_name)
        self.exists = os.path.exists(self.file_path)
        self.log = None
        self.passes = 0
        self.passed_tests = []
        self.fails = 0
        self.failed_tests = []
        self.excludes = 0
        self.excluded_tests = []
        self.test_name ='no_test'
        self.name = 'no_name'

    def print_results(self):
        total = self.passes + self.fails
        # log was empty or not valid
        if not total:
            return

        percent = self.passes/total * 100
        if (verbose):
            self.logger.log(
                f"<>{self.stage_name}: ", lpad=1, ljust=40, end_delimiter = ''
            )
        else:
            self.logger.log(
                f"{self.stage_name}: ", lpad=1, ljust=40, end_delimiter = ''
            )
        self.logger.log(f"{self.passes}:{total} ", ljust=10, end_delimiter = '')
        self.logger.log(f": {percent:.2f}% : ", ljust=12, end_delimiter = '')
        self.logger.log("Pass", end_delimiter = '')
        if (self.excludes > 0):
            self.logger.log(f"  :  {self.excludes:3.0f} : Excluded/Notrun")
        else:
            self.logger.log("")

        if (verbose and self.passes):
            self.logger.log(f"Passed tests: {self.passes}", lpad=2)
            for test in self.passed_tests:
                    self.logger.log(f'{test}', lpad=3)
        if self.fails:
            self.logger.log(f"Failed tests: {self.fails}", lpad=2)
            for test in self.failed_tests:
                    self.logger.log(f'{test}', lpad=3)
        if (verbose):
            if self.excludes:
                self.logger.log(
                    f"Excluded/Notrun tests: {self.excludes} ", lpad=2
                )
                for test in self.excluded_tests:
                    self.logger.log(f'{test}', lpad=3)

    def check_name(self, line):
        return

    def check_pass(self, line):
        return

    def check_fail(self, line):
        if "exiting with" in line:
            self.fails += 1

    def check_exclude(self, line):
        return

    def check_line(self, line):
        self.check_name(line)
        self.check_pass(line)
        self.check_fail(line)
        self.check_exclude(line)

    def read_file(self):
        with open(self.file_path, 'r') as log_file:
            for line in log_file:
                self.check_line(line.lower())

    def summarize(self):
        if not self.exists:
            return 0

        self.read_file()
        self.print_results()
        return int(self.fails)

class FiInfoSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)

    def check_fail(self, line):
        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"fi_info {self.prov}")

    def read_file(self):
        super().read_file()

        if not self.fails:
            self.passes += 1
            self.passed_tests.append(f"fi_info {self.prov}")

class FabtestsSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)

    def check_name(self, line):
        # don't double count ubertest output and don't count fi_ubertest's
        # invocation
        if 'ubertest' in line and 'client_cmd:' in line:
            self.test_name = 'no_test'
            if 'name:' not in line: # skip past client output in ubertest
                return

        test_name = line.split("name:")
        if len(test_name) > 1:
            self.test_name = test_name[-1].lower().strip()

    def get_result_line(self, line) -> Tuple[str,str]:
        result = line.split("result:")
        if len(result) > 1:
            return (result[-1].lower().strip(), line.split())
        return None, None

    def check_pass(self, line):
        result, result_line = self.get_result_line(line)
        if result == 'pass' or result == 'success' or result == 'passed':
            self.passes += 1
            if 'ubertest' in self.test_name:
                idx = (result_line.index('result:') - 1)
                ubertest_number = int((result_line[idx].split(',')[0]))
                self.passed_tests.append(f"{self.test_name}: "\
                                         f"{ubertest_number}")
            else:
                self.passed_tests.append(self.test_name)

    def check_fail(self, line):
        result, result_line = self.get_result_line(line)
        if result == 'fail':
            self.fails += 1
            if 'ubertest' in self.test_name:
                idx = (result_line.index('result:') - 1)
                ubertest_number = int((result_line[idx].split(',')[0]))
                self.failed_tests.append(f"{self.test_name}: " \
                                         f"{ubertest_number}")
            else:
                self.failed_tests.append(self.test_name)

        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(self.test_name)

    def check_exclude(self, line):
        result, _ = self.get_result_line(line)
        if result == 'excluded' or result == 'notrun':
            self.excludes += 1
            self.excluded_tests.append(self.test_name)

    def check_line(self, line):
        self.check_name(line)
        if (self.test_name != 'no_test'):
            self.check_pass(line)
            self.check_fail(line)
            self.check_exclude(line)

class MultinodePerformanceSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)

    def check_name(self, line):
        #name lines look like "starting <test_name>... <result>"
        if 'starting' in line and '...' in line:
            self.test_name = line.split()[1].split('.')[0]

    def check_pass(self, line):
        if 'pass' in line:
            self.passes += 1
            self.passed_tests.append(self.test_name)

    def check_fail(self, line):
        if 'fail' in line:
            self.fails += 1
            self.failed_tests.append(self.test_name)

        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(self.test_name)

class OnecclSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)
        self.file_path = os.path.join(self.log_dir, self.file_name)
        self.exists = os.path.exists(self.file_path)
        self.name = 'no_test'

    def check_name(self, line):
        #lines look like path/run_oneccl.sh ..... -test examples ..... test_name
        if " -test" in line:
            tokens = line.split()
            self.name = f"{tokens[tokens.index('-test') + 1]} " \
                   f"{tokens[len(tokens) - 1]}"

    def check_pass(self, line):
        if 'passed' in line or "all done" in line:
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        if 'failed' in line or "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(self.name)

class ShmemSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)
        self.shmem_type = {
            'uh'    : { 'func'      : self.check_uh,
                        'keyphrase' : 'summary',
                        'passes'    : 0,
                        'fails'     : 0
                      },
            'isx'   : { 'func'      : self.check_isx,
                        'keyphrase' : 'scaling',
                        'passes'    : 0,
                        'fails'     : 0
                      },
            'prk'   : { 'func'      : self.check_prk,
                        'keyphrase' : 'solution',
                        'passes'    : 0,
                        'fails'     : 0
                      }
        }
        self.test_type = 'prk'
        self.keyphrase = self.shmem_type[self.test_type]['keyphrase']
        self.name = 'no_test'

    def check_uh(self, line, log_file):
        # (test_002) Running test_shmem_atomics.x: Test all atomics... OK
        # (test_003) Running test_shmem_barrier.x: Tests barrier ... Failed
        if "running test_" in line:
            tokens = line.split()
            for token in tokens:
                if 'test_' in token:
                    self.name = token
            if tokens[len(tokens) - 1] == 'ok':
                self.shmem_type[self.test_type]['passes'] += 1
                self.passed_tests.append(self.name)
            else:
                self.shmem_type[self.test_type]['fails'] += 1
                self.failed_tests.append(self.name)
        # Summary
        # x/z Passed.
        # y/z Failed.
        if self.keyphrase in line: #double check
            passed = log_file.readline().lower()
            failed = log_file.readline().lower()
            token = int(passed.split()[1].split('/')[0])
            if self.shmem_type[self.test_type]['passes'] != token:
                self.logger.log(
                    f"passes {self.shmem_type[self.test_type]['passes']} do " \
                    f"not match log reported passes {token}"
                )
            token = int(failed.split()[1].split('/')[0])
            if self.shmem_type[self.test_type]['fails'] != int(token):
                self.logger.log(
                    f"fails {self.shmem_type[self.test_type]['fails']} does "\
                    f"not match log fails {token}"
                )

    def check_prk(self, line, log_file=None):
        if self.keyphrase in line:
            self.shmem_type[self.test_type]['passes'] += 1
        if 'error:' in line or "exiting with" in line:
            self.shmem_type[self.test_type]['fails'] += 1
            p = self.shmem_type[self.test_type]['passes']
            f = self.shmem_type[self.test_type]['fails']
            self.failed_tests.append(f"{self.prov} {p + f}")
        if 'test(s)' in line:
            token = line.split()[0]
            if self.fails != int(token):
                self.logger.log(
                    f"fails {self.fails} does not match log reported fails " \
                    f"{token}"
                )

    def check_isx(self, line, log_file=None):
        if self.keyphrase in line:
            self.shmem_type[self.test_type]['passes'] += 1
        if ('failed' in line and 'test(s)' not in line) or \
            "exiting with" in line:
            self.shmem_type[self.test_type]['fails'] += 1
            p = self.shmem_type[self.test_type]['passes']
            f = self.shmem_type[self.test_type]['fails']
            self.failed_tests.append(f"{self.prov} {p + f}")
        if 'test(s)' in line:
            token = line.split()[0]
            if int(token) != self.shmem_type[self.test_type]['fails']:
                self.logger.log(
                    f"fails {self.shmem_type[self.test_type]['fails']} does " \
                    f"not match log reported fails {int(token)}"
                )

    def check_fails(self, line):
        if "exiting with" in line:
            self.shmem_type[self.test_type]['fails'] += 1
            p = self.shmem_type[self.test_type]['passes']
            f = self.shmem_type[self.test_type]['fails']
            self.failed_tests.append(f"{self.prov} {p + f}")

    def check_test_type(self, line):
        if "running shmem" in line:
            self.test_type = line.split(' ')[2].lower()
            self.keyphrase = self.shmem_type[self.test_type]['keyphrase']

    def check_line(self, line, log_file):
        self.check_test_type(line)
        if self.test_type is not None:
            self.shmem_type[self.test_type]['func'](line, log_file)
            self.check_fails(line)

    def read_file(self):
        with open(self.file_path, 'r') as log_file:
            for line in log_file:
                self.check_line(line.lower(), log_file)

        for key in self.shmem_type.keys():
            self.passes += self.shmem_type[key]['passes']
            self.fails += self.shmem_type[key]['fails']

class MpichTestSuiteSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, mpi, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)

        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = '/mpiexec'
        else:
            self.run = '/mpirun'

    def check_name(self, line):
        if self.run in line:
            self.name = line.split()[len(line.split()) - 1].split('/')[1]
            #assume pass
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        # Fail cases take away assumed pass
        if "exiting with" in line:
            self.fails += 1
            self.passes -= 1
            self.failed_tests.append(f'{self.name}')
            #skip to next test
            while self.run not in line:
                line = self.log.readline().lower()

class ImbSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, mpi, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)

        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = 'mpiexec'
        else:
            self.run = 'mpirun'
        self.test_type = ''

    def check_type(self, line):
        if 'part' in line:
            self.test_type = line.split()[len(line.split()) - 2]

    def check_name(self, line):
        if "benchmarking" in line:
            self.name = line.split()[len(line.split()) - 1]

    def check_pass(self, line):
        if "benchmarking" in line:
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"{self.test_type} {self.name}")
            self.passes -= 1

    def check_line(self, line):
        self.check_type(line)
        self.check_name(line)
        self.check_pass(line)
        self.check_fail(line)
        super().check_exclude(line)

class OsuSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, mpi, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)
        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = 'mpiexec'
        else:
            self.run = 'mpirun'

        self.type = ''
        self.tokens = []

    def get_tokens(self, line):
        if "# osu" in line:
            self.tokens = line.split()
        else:
            self.tokens = []

    def check_name(self, line):
        if 'osu' in self.tokens:
            self.name = " ".join(self.tokens[self.tokens.index('osu') + \
                        1:self.tokens.index('test')])

    def check_type(self):
        if self.tokens:
            self.test_type = self.tokens[1]

    def check_pass(self, line):
        if 'osu' in self.tokens:
            # Assume pass
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"{self.test_type} {self.name}")
            # Remove assumed pass
            self.passes -= 1

    def check_line(self, line):
        self.get_tokens(line)
        self.check_name(line)
        self.check_type()
        self.check_pass(line)
        self.check_fail(line)
        super().check_exclude(line)

class DaosSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)

    def check_name(self, line):
        if "reading ." in line:
            self.test_name = line.split('/')[len(line.split('/')) - 1] \
                             .rstrip('.yaml\n')

    def check_pass(self, line):
        res_string = line.lstrip("results    :").rstrip()
        res_list = res_string.split(' | ')
        for elem in res_list:
            if 'pass' in elem:
                self.passes += [int(s) for s in elem.split() if s.isdigit()][0]
                display_testname = self.test_name.ljust(20)
                self.passed_tests.append(f"{display_testname} : {res_string}")

    def check_fail(self, line):
        res_list = line.lstrip("results    :").rstrip().split('|')
        for elem in res_list:
            if 'pass' not in elem:
                self.fails += [int(s) for s in elem.split() if s.isdigit()][0]
                if self.fails != 0:
                    self.failed_tests.append(f'{self.test_name}')
        return (self.fails)

    def check_line(self, line):
        self.check_name(line)
        if "results    :" in line:
            self.check_pass(line)
            self.check_fail(line)

def get_release_num(log_dir):
    file_name = f'{log_dir}/release_num.txt'
    if os.path.exists(file_name):
        with open(file_name) as f:
            num = f.readline()

        return num.strip()

    raise Exception("No release num")

def summarize_items(summary_item, logger, log_dir, mode):
    err = 0
    mpi_list = ['impi', 'mpich', 'ompi']
    logger.log(f"Summarizing {mode} build mode:")
    if summary_item == 'fabtests' or summary_item == 'all':
        for prov,util in common.prov_list:
            if util:
                prov = f'{prov}-{util}'
            ret = FabtestsSummarizer(
                logger, log_dir, prov,
                f'{prov}_fabtests_{mode}',
                f"{prov} fabtests {mode}"
            ).summarize()
            err += ret if ret else 0
            ret = FiInfoSummarizer(
                logger, log_dir, prov,
                f'{prov}_fi_info_{mode}',
                f"{prov} fi_info {mode}"
            ).summarize()
            err += ret if ret else 0

    if summary_item == 'imb' or summary_item == 'all':
        for mpi in mpi_list:
            for item in ['tcp-rxm', 'verbs-rxm', 'tcp']:
                ret = ImbSummarizer(
                    logger, log_dir, item, mpi,
                    f'MPI_{item}_{mpi}_IMB_{mode}',
                    f"{item} {mpi} IMB {mode}"
                ).summarize()
                err += ret if ret else 0

    if summary_item == 'osu' or summary_item == 'all':
        for mpi in mpi_list:
                for item in ['tcp-rxm', 'verbs-rxm']:
                    ret = OsuSummarizer(
                        logger, log_dir, item, mpi,
                        f'MPI_{item}_{mpi}_osu_{mode}',
                        f"{item} {mpi} OSU {mode}"
                    ).summarize()
                    err += ret if ret else 0

    if summary_item == 'mpichtestsuite' or summary_item == 'all':
        for mpi in mpi_list:
            for item in ['tcp-rxm', 'verbs-rxm', 'sockets']:
                ret = MpichTestSuiteSummarizer(
                    logger, log_dir, item, mpi,
                    f'mpichtestsuite_{item}_{mpi}_'\
                    f'mpichtestsuite_{mode}',
                    f"{item} {mpi} mpichtestsuite {mode}"
                ).summarize()
                err += ret if ret else 0
    if summary_item == 'multinode' or summary_item == 'all':
        for prov,util in common.prov_list:
            if util:
                prov = f'{prov}-{util}'

            ret = MultinodePerformanceSummarizer(
                logger, log_dir, prov,
                f'multinode_performance_{prov}_{mode}',
                f"multinode performance {prov} {mode}"
            ).summarize()
            err += ret if ret else 0

    if summary_item == 'oneccl' or summary_item == 'all':
        for prov in ['tcp-rxm', 'verbs-rxm']:
            ret = OnecclSummarizer(
                logger, log_dir, 'oneCCL',
                f'oneCCL_{prov}_oneccl_{mode}',
                f'oneCCL {prov} {mode}'
            ).summarize()
            err += ret if ret else 0
            ret = OnecclSummarizer(
                logger, log_dir, 'oneCCL-GPU',
                f'oneCCL-GPU_{prov}_onecclgpu_{mode}',
                f'oneCCL-GPU {prov} {mode}'
            ).summarize()
        err += ret if ret else 0

    if summary_item == 'shmem' or summary_item == 'all':
        for prov in ['tcp', 'verbs', 'sockets']:
            ret= ShmemSummarizer(
                logger, log_dir, prov,
                f'SHMEM_{prov}_shmem_{mode}',
                f'shmem {prov} {mode}'
            ).summarize()
        err += ret if ret else 0

    if summary_item == 'ze' or summary_item == 'all':
        test_types = ['h2d', 'd2d', 'xd2d']
        for type in test_types:
            for prov in ['shm']:
                ret = FabtestsSummarizer(
                    logger, log_dir, 'shm',
                    f'ze_{prov}_{type}_{mode}',
                    f"ze {prov} {type} {mode}"
                ).summarize()
                err += ret if ret else 0

    if ((summary_item == 'daos' or summary_item == 'all')
         and mode == 'reg'):
        for prov in ['tcp-rxm', 'verbs-rxm']:
            ret = DaosSummarizer(
                logger, log_dir, prov,
                f'daos_{prov}_{mode}',
                f"{prov} daos {mode}"
            ).summarize()
            err += ret if ret else 0

    return err

if __name__ == "__main__":
#read Jenkins environment variables
    # In Jenkins,  JOB_NAME  = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
    # job name is better to use to distinguish between builds of different
    # jobs but with same branch name.
    jobname = os.environ['JOB_NAME']
    buildno = os.environ['BUILD_NUMBER']
    workspace = os.environ['WORKSPACE']

    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_item', help="functional test to summarize",
                         choices=['fabtests', 'imb', 'osu', 'mpichtestsuite',
                         'oneccl', 'shmem', 'ze', 'multinode', 'daos', 'all'])
    parser.add_argument('--ofi_build_mode', help="select buildmode debug or dl",
                        choices=['dbg', 'dl', 'reg'], default='all')
    parser.add_argument('-v', help="Verbose mode. Print all tests", \
                        action='store_true')
    parser.add_argument('--release', help="This job is testing a release."\
                        "It will be saved and checked into a git tree.",
                        action='store_true')

    args = parser.parse_args()
    verbose = args.v
    summary_item = args.summary_item
    release = args.release
    ofi_build_mode = args.ofi_build_mode

    mpi_list = ['impi', 'mpich', 'ompi']
    log_dir = f'{cloudbees_config.install_dir}/{jobname}/{buildno}/log_dir'

    if (release):
        release_num = get_release_num(log_dir)
        job_name = os.environ['JOB_NAME'].replace('/', '_')
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = f'summary_{release_num}_{job_name}_{date}.log'
        full_file_name = f'{log_dir}/{output_name}'
    else:
        full_file_name = NamedTemporaryFile(prefix="summary.out.").name

    with open(full_file_name, 'a') as output_file:
        if (ofi_build_mode == 'all'):
            output_file.truncate(0)

        logger = Logger(output_file, release)
        if (release):
            Release(
                log_dir, output_file, logger, release_num
            ).add_release_changes()

        err = 0
        build_modes = ['reg', 'dbg', 'dl']
        for mode in build_modes:
            if ofi_build_mode != 'all' and mode != ofi_build_mode:
                continue

            err += summarize_items(summary_item, logger, log_dir, mode)

    if (release):
        shutil.copyfile(f'{full_file_name}', f'{workspace}/{output_name}')

    exit(err)
