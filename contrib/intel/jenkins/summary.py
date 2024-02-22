from abc import ABC, abstractmethod
import shutil
from datetime import datetime
from typing import Tuple
import os
from pickle import FALSE
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# add jenkins config location to PATH
sys.path.append(f"{os.environ['CUSTOM_WORKSPACE']}/ci_resources/configs/{os.environ['CLUSTER']}")

import cloudbees_config
import argparse
import common

verbose = False

class SendEmail:
    def __init__(self, sender=None, receivers=None, attachment=None):
        self.sender = sender if sender is not None else os.environ['SENDER']
        self.receivers = (receivers if receivers is not None else \
                         f"{os.environ['RECEIVER']}").split(',')
        self.attachment = attachment
        self.work_week = datetime.today().isocalendar()[1]
        self.msg = MIMEMultipart()

    def __add_attachments(self):
        print(f"Attachment is {self.attachment}")
        if self.attachment is None:
            return

        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(open(self.attachment, 'rb').read())
        encoders.encode_base64(attachment)
        name = f"Jenkins_Summary_ww{self.work_week}"
        if (verbose):
            name = f"{name}_all"
        attachment.add_header('Content-Disposition',
                                f"attachment; filename={name}")
        self.msg.attach(attachment)

    def __write_msg(self):
        self.msg['Subject'] = f"Cloudbees Summary {os.environ['JOB_NAME']}"
        self.msg['From'] = self.sender
        self.msg['To'] = ", ".join(self.receivers)
        self.msg.attach(MIMEText(f"WW{self.work_week} Summary for Libfabric "\
                                 "From Cloudbees"))

    def send_mail(self):
        self.__write_msg()
        self.__add_attachments()
        server = smtplib.SMTP(os.environ['SMTP_SERVER'],
                              os.environ['SMTP_PORT'])
        server.sendmail(self.sender, self.receivers, self.msg.as_string())
        server.quit()

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
        self.output_file.write(f'{self.padding * lpad}{line}{end_delimiter}')

class Summarizer(ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "print_results")
            and callable(subclass.print_results)
            and hasattr(subclass, "check_features")
            and callable(subclass.check_features)
            and hasattr(subclass, "check_node")
            and callable(subclass.check_node)
            and hasattr(subclass, "check_name")
            and callable(subclass.check_name)
            and hasattr(subclass, "check_pass")
            and callable(subclass.check_pass)
            and hasattr(subclass, "check_fail")
            and callable(subclass.check_fail)
            and hasattr(subclass, "check_exclude")
            and callable(subclass.check_exclude)
            and hasattr(subclass, "fast_forward")
            and callable(subclass.fast_forward)
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
        self.error = 0
        self.errored_tests = []
        self.test_name ='no_test'
        self.name = 'no_name'
        self.features = "no_features_found"
        self.node = "no_node_found"

    def print_results(self):
        total = self.passes + self.fails
        # log was empty or not valid
        if not total:
            return

        percent = self.passes/total * 100
        if (verbose):
            self.logger.log(
                f"<>{self.stage_name} : ", lpad=1, ljust=50, end_delimiter = ''
            )
        else:
            self.logger.log(
                f"{self.stage_name} : ",
                lpad=1, ljust=50, end_delimiter = ''
            )
        self.logger.log(
                f"{self.node} : ",
                lpad=1, ljust=20, end_delimiter = ''
        )
        self.logger.log(
                f"[{self.features}] : ",
                lpad=1, ljust=30, end_delimiter = ''
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

            if self.error:
                self.logger.log(
                    "Errored, Interrupt, or Canceled Tests: "\
                    f"{self.excludes} ", lpad=2
                )
                for test in self.errored_tests:
                    self.logger.log(f'{test}', lpad=3)

    def check_features(self, previous, line):
        if ('avail_features') in previous:
            self.features = line.strip()

    def check_node(self, line):
        if ('slurm_nodelist' in line):
            self.node = line.strip().split('=')[1]

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

    def fast_forward(self, log_file):
        previous = ""
        line = log_file.readline().lower()
        while line != "":
            self.check_node(line)
            self.check_features(previous, line)
            if common.cloudbees_log_start_string.lower() in line:
                break

            previous = line
            line = log_file.readline().lower()

    def read_file(self):
        with open(self.file_path, 'r') as log_file:
            self.fast_forward(log_file)
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
        self.trace = False

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
                try:
                    int((result_line[idx].split(',')[0]))
                except:
                    return

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
                try:
                    int((result_line[idx].split(',')[0]))
                except:
                    return
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

    def check_trace(self, line):
        if not self.trace:
            cmd_count = 0
            faults_count = 0
            if ("user to sar buffer" in line):
                tokens = line.split(' ')
                for i in range(0, len(tokens)):
                    if 'cmd' in tokens[i]:
                        cmd_count += int(tokens[i + 1])
                    if 'faults' in tokens[i]:
                        faults_count += int(tokens[i + 1])

                if (cmd_count > 0 or faults_count > 0):
                    self.trace = True

    def check_line(self, line):
        self.check_name(line)
        if (self.test_name != 'no_test'):
            self.check_pass(line)
            self.check_fail(line)
            self.check_exclude(line)
            if ('dsa' in self.file_name):
                self.check_trace(line)

    def summarize(self):
        if not self.exists:
            return 0

        self.read_file()
        self.print_results()
        if ('dsa' in self.file_name and not self.trace):
            exit("Expected: DSA to run. Actual: DSA Not Run")

        return int(self.fails)

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

    def read_file(self):
        with open(self.file_path, 'r') as log_file:
            self.fast_forward(log_file)
            for line in log_file:
                self.check_line(line)

    def check_name(self, line):
        #OneCCL GPU tests:
        if "bash -c" in line and "./run.sh" not in line:
            tokens = line.split('./')[1]
            self.name = tokens.split()[0]
        #OneCCL CPU tests:
        if "Running" in line and "CCL_LOG_LEVEL=debug" not in line:
            if './' in line:
                tokens = line.split('./')[1]
                self.name = tokens.split()[0]

    def check_pass(self, line):
        if '[0] PASSED' in line or "All done" in line:
            self.passes += 1
            self.passed_tests.append(f"{self.name}: 1")
        if ("[0] [  PASSED  ]" in line and "tests." in line) or \
            ("tests." in line and "[1] [  PASSED  ]" not in line and \
            "[0] [  PASSED  ]" not in line):
            token = line.split()
            no_of_tests = f"{token[token.index('tests.') - 1]} "
            self.passes += int(no_of_tests)
            self.passed_tests.append(f"{self.name}: {no_of_tests}")

    def check_fail(self, line):
        if 'failed' in line or "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(self.name)

class ShmemSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)
        self.name = 'no_test'

    def check_name(self, line):
        line = line.strip()
        if "running " in line:
            tokens = line.split(' ')
            self.name = ' '.join(tokens[1:])

    def check_pass(self, line):
        line = line.strip()
        if "pass!" in line:
            self.passes += 1
            self.passed_tests.append(self.name)


class MpichTestSuiteSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, mpi, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)

        self.mpi = mpi
        self.run = 'mpiexec'
    
    def read_file(self):
        with open(self.file_path,'r') as log_file:
            super().fast_forward(log_file)
            for line in log_file:
                super().check_line(line.lower().strip())

    def check_exclude(self, line):
        if line.startswith('excluding:'):
            test = line.split(':')[-1]
            self.excludes += 1
            self.excluded_tests.append(test)

    def check_name(self, line):
        if (line.startswith('ok') or 
            line.startswith('not ok')):
                self.name = line.split('-')[1].split('#')[0].strip()

    def check_pass(self, line):
        if (line.startswith('ok') and not
            line.split('#')[1].strip().startswith('skip')):
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        if (line.startswith('not ok') and not
            line.split('#')[1].strip().startswith('skip')):
            self.fails += 1
            self.failed_tests.append(self.name)


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

        if (self.exists):
            if ('verbs' in file_name):
                self.node = cloudbees_config.daos_prov_node_map['verbs']
            if ('tcp' in file_name):
                self.node = cloudbees_config.daos_prov_node_map['tcp']

            self.features = cloudbees_config.daos_node_features

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
            total = [int(s) for s in elem.split() if s.isdigit()][0]
            if total != 0:
                if 'fail' in elem:
                    self.fails += total
                    self.failed_tests.append(f'{self.test_name}')
                if 'error' in elem:
                    self.error += total
                    self.errored_tests.append(f'error: {self.test_name}')
                if 'interrupt' in elem:
                    self.error += total
                    self.errored_tests.append(f'interrupt: {self.test_name}')
                if 'cancel' in elem:
                    self.error += total
                    self.errored_tests.append(f'cancel: {self.test_name}')
    
    def check_exclude(self, line):
        res_list = line.lstrip("results    :").rstrip().split('|')
        for elem in res_list:
            total = [int(s) for s in elem.split() if s.isdigit()][0]
            if total != 0:
                if 'skip' in elem:
                    self.excludes += total
                    self.excluded_tests.append(f'skip: {self.test_name}')
                if 'warn' in elem:
                    self.excludes += total
                    self.excluded_tests.append(f'warn: {self.test_name}')

    def check_line(self, line):
        self.check_name(line)
        if "results    :" in line:
            self.check_pass(line)
            self.check_fail(line)
            self.check_exclude(line)

class DmabufSummarizer(Summarizer):
    def __init__(self, logger, log_dir, prov, file_name, stage_name):
        super().__init__(logger, log_dir, prov, file_name, stage_name)

        self.test_type = ''

    def check_type(self, line):
        if "Running" in line:
            self.test_type = line.split()[2]

    def check_num_node(self, line):
        if "SLURM_NNODES" in line:
            self.num_nodes = line.split("=")[-1].strip()
            self.num_nodes = ' '.join([self.num_nodes, 'node'])

    def check_name(self, line):
        if "client_command" in line:
            name_list = line.split()[-2:]
            name_list.insert(0, str(self.num_nodes))
            name_list.insert(1, str(self.test_type))
            self.test_name = name_list

    def check_pass(self, line):
        if "TEST COMPLETED" in line:
            self.passes += 1
            self.passed_tests.append(self.test_name)

    def check_fail(self, line):
        if "TEST FAILED" in line:
            self.fails += 1
            self.failed_tests.append(self.test_name)

    def fast_forward(self, log_file):
        previous = ""
        line = log_file.readline()
        while line != "":
            self.check_num_node(line)
            self.check_node(line.lower())
            self.check_features(previous.lower(), line.lower())
            if common.cloudbees_log_start_string.lower() in line.lower():
                break

            previous = line
            line = log_file.readline()

    def read_file(self):
        with open(self.file_path, 'r') as log_file:
            self.fast_forward(log_file)
            for line in log_file:
                self.check_type(line)
                self.check_line(line)

def get_release_num():
    file_name = f'{os.environ["CUSTOM_WORKSPACE"]}/source/libfabric/'\
                'release_num.txt'
    if os.path.exists(file_name):
        with open(file_name) as f:
            num = f.readline()

        return num.strip()

    raise Exception("No release num")

def summarize_items(summary_item, logger, log_dir, mode):
    err = 0
    mpi_list = ['impi', 'mpich', 'ompi']
    logger.log(f"Summarizing {mode} build mode:")
    provs = common.prov_list + [('tcp-iouring', None)]
    if summary_item == 'fabtests' or summary_item == 'all':
        for prov,util in provs:
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

    if ((summary_item == 'daos' or summary_item == 'all')
         and mode == 'reg'):
        for prov in ['tcp-rxm', 'verbs-rxm']:
            ret = DaosSummarizer(
                logger, log_dir, prov,
                f'daos_{prov}_{mode}',
                f"{prov} daos {mode}"
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
                for item in ['tcp-rxm', 'verbs-rxm', 'tcp']:
                    ret = OsuSummarizer(
                        logger, log_dir, item, mpi,
                        f'MPI_{item}_{mpi}_osu_{mode}',
                        f"{item} {mpi} OSU {mode}"
                    ).summarize()
                    err += ret if ret else 0

    if summary_item == 'mpichtestsuite' or summary_item == 'all':
        for mpi in mpi_list:
            for item in ['tcp', 'verbs-rxm']:
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
                f'multinode_performance_{prov}_multinode_{mode}',
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
        for prov in ['tcp', 'verbs-rxm', 'sockets']:
            ret= ShmemSummarizer(
                logger, log_dir, prov,
                f'SHMEM_{prov}_shmem_{mode}',
                f'shmem {prov} {mode}'
            ).summarize()
        err += ret if ret else 0

    if summary_item == 'v3' or summary_item == 'all':
        test_types = ['h2d', 'd2d', 'xd2d']
        for t in test_types:
            ret = FabtestsSummarizer(
                logger, log_dir, 'shm',
                f'ze_v3_shm_{t}_fabtests_{mode}',
                f"ze v3 shm {t} fabtests {mode}"
            ).summarize()
            err += ret if ret else 0

        ret = OnecclSummarizer(
                logger, log_dir, 'oneCCL-GPU',
                f'oneCCL-GPU-v3_verbs-rxm_onecclgpu_{mode}',
                f'oneCCL-GPU-v3 verbs-rxm {mode}'
        ).summarize()
        err += ret if ret else 0

    if summary_item == 'dsa' or summary_item == 'all':
        for prov in ['shm']:
            ret = FabtestsSummarizer(
                logger, log_dir, 'shm',
                f'{prov}_dsa_fabtests_{mode}',
                f"{prov} dsa fabtests {mode}"
            ).summarize()
            err += ret if ret else 0

    if summary_item == 'dmabuf' or summary_item == 'all':
        for prov in ['verbs-rxm']:
            for num_nodes in range(1,3):
                ret = DmabufSummarizer(
                    logger, log_dir, 'verbs-rxm',
                    f'DMABUF-Tests_{prov}_dmabuf_{num_nodes}_{mode}',
                    f"DMABUF-Tests {prov} dmabuf {num_nodes} node {mode}"
                ).summarize()
                err += ret if ret else 0

    if summary_item == 'cuda' or summary_item == 'all':
        test_types = ['h2d', 'd2d', 'xd2d']
        for v in range(1, 3):
            for t in test_types:
                ret = FabtestsSummarizer(
                    logger, log_dir, 'shm',
                    f'cuda_v{v}_shm_{t}_fabtests_{mode}',
                    f"cuda v{v} shm {t} fabtests {mode}"
                ).summarize()
            err += ret if ret else 0

    return err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_item', help="functional test to summarize",
                         choices=['fabtests', 'imb', 'osu', 'mpichtestsuite',
                         'oneccl', 'shmem', 'multinode', 'daos', 'v3',
                         'dsa', 'dmabuf', 'all'])
    parser.add_argument('--ofi_build_mode', help="select buildmode debug or dl",
                        choices=['dbg', 'dl', 'reg'], default='all')
    parser.add_argument('-v', help="Verbose mode. Print all tests", \
                        action='store_true')
    parser.add_argument('--release', help="This job is testing a release."\
                        "It will be saved and checked into a git tree.",
                        action='store_true')
    parser.add_argument('--send_mail', help="Email mailing list with summary "\
                        "results", action='store_true')

    args = parser.parse_args()
    verbose = args.v
    summary_item = args.summary_item
    release = args.release
    ofi_build_mode = args.ofi_build_mode
    send_mail = args.send_mail

    mpi_list = ['impi', 'mpich', 'ompi']
    custom_workspace = os.environ['CUSTOM_WORKSPACE']
    log_dir = f'{custom_workspace}/log_dir'
    if (not os.path.exists(log_dir)):
        os.makedirs(log_dir)

    job_name = os.environ['JOB_NAME'].replace('/', '_')

    print(f"Files to be summarized: {os.listdir(log_dir)}")

    if (release):
        release_num = get_release_num()
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = f'summary_{release_num}_{job_name}_{date}.log'
    else:
        output_name = f'summary_{job_name}.log'

    full_file_name = f'{log_dir}/{output_name}'

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
        shutil.copyfile(f'{full_file_name}', f'{custom_workspace}/{output_name}')

    if (send_mail):
        SendEmail(sender = os.environ['SENDER'],
                  receivers = os.environ['mailrecipients'],
                  attachment = full_file_name
                 ).send_mail()

    exit(err)
