#! /bin/sh

if test ! -f src/udpx.h; then
    echo You really need to run this script in the prov udp directory
    exit 1
fi

set -x

autoreconf -ivf
