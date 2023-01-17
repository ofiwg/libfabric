#!/bin/bash
git checkout -b rebase-test-branch
./autogen.sh
PKG_CONFIG_PATH=$(realpath ../libcxi/install/lib/pkgconfig/) ./configure \
	--disable-memhooks-monitor \
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
	--with-criterion=$(realpath ../Criterion/build/install) 

rm -f commit-results.tap
mb=$(git merge-base origin/v1.15.x-ss11 HEAD)
git rebase ${mb} --exec 'set -e; git log -1; ID=$(git rev-parse HEAD); TESTS="not ok"; COMMIT="not ok"; make clean && make -j && ssh -tt localhost "cd ~/workspace/workspace/os-networking-team/cassini-vm/PR-libfabric/prov/cxi/test && ./run_tests_vm.sh"; ECODE=1; grep -q "not ok" prov/cxi/test/cxitest.tap || ECODE=$(($?^1)); [[ "$ECODE" -eq "0" ]] && TESTS="ok"; echo "${TESTS} - Commit ${ID} Tests Status" >> commit-results.tap; git log -1 | grep -q "Signed-off-by: "; [[ "$?" -eq "0" ]] && COMMIT="ok"; echo "${COMMIT} - Commit ${ID} Signed-off" >> commit-results.tap; mv prov/cxi/test/cxitest.out prov/cxi/test/${ID}-cxitest.out; exit $ECODE'
sed -i -e "1i TAP version 13\n1..$(wc -l commit-results.tap | cut -c1)" commit-results.tap
mv -f commit-results.tap prov/cxi/test/
