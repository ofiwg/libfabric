#!/bin/bash

test_short=""
exclude_commit_subject=""

while getopts "se:" option; do
	case "${option}" in
		s)
			test_short="-s"
			;;
		e)
			exclude_commit_subject=${OPTARG}
			;;
		*)
			exit 1;
	esac
done

# Assumes that the commit subject is unique between all commits in the PR.
head_commit_subject_collapsed=$(git log -1 --pretty=%s | tr -d ' ')
if [[ "$head_commit_subject_collapsed" == "$exclude_commit_subject" ]]; then
	echo "Skippping commit \"$(git log -1 --pretty=%s)\""
	exit 0
fi

git log -1

set -e

cd ../../../
./autogen.sh
./configure \
	--prefix=$PWD/install \
	--disable-sockets \
	--disable-udp \
	--disable-verbs \
	--disable-rxm \
	--disable-mrail \
	--disable-rxd \
	--disable-shm \
	--disable-tcp \
	--disable-usnic \
	--disable-rstream \
	--disable-efa \
	--disable-psm2 \
	--disable-psm3 \
	--disable-opx \
	--enable-debug \
	--with-default-monitor=uffd \
	--with-criterion=$(realpath ../Criterion/build/install/) \
	--with-cassini-headers=$(realpath ../cassini-headers/install) \
	--with-cxi-uapi-headers=$(realpath ../cxi-driver) \
	--enable-cxi=$(realpath ../libcxi/install) \
	--with-kdreg2=$(realpath ../kdreg2/include)


make clean
make -j 8 install

test_dir=$(realpath ./prov/cxi/test)
test_result_file="run_tests_vm_output.txt"
ssh -tt localhost "cd ${test_dir}; ./run_tests_vm.sh $test_short" | tee ${test_result_file}

set +e

# Search ssh output for the following string. This is a test failure
# which is not reported as a tap test failure.
test_error_code=1
test_error=$(grep "cxitest return non-zero exit code. Possible failures in test teardown" ${test_result_file}) || test_error_code=$(($?^1))
if [ -z "${test_error}" ] && [ "$test_error_code" -eq "0" ]; then
	echo "Zero 'non-zero exit codes' failures in output"
else
	echo $test_error
	exit 1
fi

# Grep all tap out files for "not ok" string. This is a test failure.
test_failures_code=1
test_failures=$(grep "not ok" ${test_dir}/*.tap) || test_failures_code=$(($?^1))
if [ -z "${test_failures}" ] && [ "$test_failures_code" -eq "0" ] ; then
	echo "Zero 'not ok' failures in tap output"
else
	echo $test_failures
	exit 1
fi

signed_off=$(git log -1 | grep -i "Signed-off-by: ")
if [ -z "${signed_off}" ]; then
	echo "Commit not signed off"
	exit 1
else
	echo "Commit signed-off check passed"
fi

echo "Tests passed"
rm ${test_result_file}
exit 0
