#!/usr/bin/env bash
# Post-build hook to copy Module.symvers into the DKMS tree alongside the .ko
# files, so that other modules can build against it
 
if [ ${#} -ne 2 ]
then
    echo "usage: dkms.post_build.sh <symvers_directory> <package_directory>"
    exit 1
fi
 
SYMVERS_DIR=${1}
PACKAGE_DIR=${2}
 
if [ ! -d ${SYMVERS_DIR} ]
then
    echo "error: ${SYMVERS_DIR} is not a directory"
    exit 1
fi
 
if [ ! -d ${PACKAGE_DIR} ]
then
    echo "error: ${PACKAGE_DIR} is not a directory"
    exit 1
fi
 
if [ ! -f ${SYMVERS_DIR}/Module.symvers ]
then
    echo "error: ${SYMVERS_DIR}/Module.symvers is not present or is not a file"
    exit 1
fi
 
if [ -z "${kernelver}" ]
then
    echo "error: '${kernelver}' is empty"
    exit 1
fi
 
if [ -z "${arch}" ]
then
    echo "error: '${arch}' is empty"
    exit 1
fi
 
if [ ! -d ${PACKAGE_DIR}/${kernelver}/${arch}/module ]
then
    echo "error: ${PACKAGE_DIR}/${kernelver}/${arch}/module directory does not exist"
    exit 1
fi
 
cp -f ${SYMVERS_DIR}/Module.symvers ${PACKAGE_DIR}/${kernelver}/${arch}/module
exit ${?}
