#! /bin/sh

if test ! -f src/psmx3.h; then
	echo You really need to run this script in the prov psm3 directory
	exit 1
fi

if [ -f psm3/Makefile.include.base ]
then
	make -f - <<EOF
psm3/Makefile.include: psm3/Makefile.include.base
	cp psm3/Makefile.include.base psm3/Makefile.include
EOF
fi

set -x

autoreconf -ivf
