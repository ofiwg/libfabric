#! /bin/sh

if test ! -d .git && test ! -f simple/info.c; then
    echo You really need to run this script in the top-level fabtests directory
    exit 1
fi

set -x

autoreconf -ivf
