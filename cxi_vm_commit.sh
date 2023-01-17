#!/bin/bash
./autogen.sh
PKG_CONFIG_PATH=$(realpath ../libcxi/install/lib/pkgconfig/) ./configure \
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
	--disable-memhooks-monitor \
	--with-criterion=$(realpath ../Criterion/build/install/)

make -j install
ssh -tt localhost "cd ~/workspace/workspace/os-networking-team/cassini-vm/libfabric/prov/cxi/test && ./run_tests_vm.sh"
