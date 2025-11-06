#!/bin/bash -x
# Copyright 2019 Cray Inc. All Rights Reserved.

# Modified copy of
# https://github.hpe.com/hpe/hpc-dst-jenkins-shared-library/blob/master/resources/rpm_build.sh
# taken from commit 153e918fdd802df2f7e7c7690f1e0cad1f0c55e5. Modified by
# richard.halkyard@hpe.com to support building for multiple kernels in one job.
# Modified area marked by '**********''

. vars.sh

# We need values for the app name, version, and architecture. Figure out
# defaults automatically.
cat >> vars.sh <<- END
ARCH=$TARGET_ARCH
BUILDNAME=noarch
OS=$TARGET_OS
SPECFILE=`ls *.spec | head -n 1`
END

. vars.sh
. zypper-local.sh

# Fanout parameters are passed in from the command line. We have to
# parse them out and re-package them because of the way bash handles
# quotes.
for KEYVAL in $@; do
    KEY=`echo $KEYVAL | cut -d '=' -f 1`
    VAL=`echo $KEYVAL | cut -d '=' -f 2`
    # If given a blank, stick with the default.
    if [ "$VAL" == "" ]; then continue; fi
    case $KEY in
        _arch)      ARCH=$VAL ;;
        _buildname) BUILDNAME=$VAL ;;
        _os)        OS=$VAL ;;
        _specfile)  SPECFILE=$VAL ;;
        *)          echo "WARNING: Unrecognized input $KEYVAL" ;;
    esac
done

SPECFILE_BASENAME=$(basename "${SPECFILE}")

set_rpm_vars()  {
    TOPDIR=$PWD/WORKSPACE
    DEBUG_PACKAGE="%{nil}"
    SOURCE_PAYLOAD="w0.gzdio"
    BINARY_PAYLOAD="w9.bzdio"
    APP="cray-cxi-driver"
}

set_rpm_dir() {
    # Support non-default archive names for calls to %setup -n <arcname>
    RPM_DIR="$(rpmspec --srpm -P ${SPECFILE} | awk '/^%setup/{ for(i=1;i<=2;i++) if ($i == "-n") print $(i+1) }')/"
    if [ "$RPM_DIR" == "/" ]; then
        RPM_DIR="$(rpmspec --srpm -q --qf "%{name}-%{version}" ${SPECFILE})"
    fi
}

GPU_ARGS=" --with nvidiagpu "

if [[ $ARCH == x86_64 ]]; then
    GPU_ARGS+=" --with amdgpu "
fi

APP_MACRO=$(grep -F "%{_app}" ${SPECFILE})
APP_MACRO=$?
if grep -F "%{_app}" ${SPECFILE}; then
    NAME=`echo $SPECFILE | rev | cut -d '.' -f 2- | rev`
    set_rpm_vars
    SOURCE=`rpmspec --srpm -q --qf "[%{source}]\n" $NAME.spec | xargs basename`
    set_rpm_dir
else
    SOURCE="$(rpmspec --srpm -q --qf "[%{source};]\n" ${SPECFILE}  | grep -o '[^;]*tar[^;]*')"
    NAME="$(rpmspec --srpm -q --qf "[%{name}]\n" ${SPECFILE})"
    set_rpm_vars
    set_rpm_dir
fi

echo "$0: --> APP_MACRO: '${APP_MACRO}'"
echo "$0: --> NAME: '${NAME}'"
echo "$0: --> SOURCE: '${SOURCE}'"

VERSION=`rpmspec --srpm -q --qf "[%{version}]\n" ${SPECFILE}`

# If a hard requirement is missing, bail.
if [ ! -f "$SPECFILE" ]; then
    echo "Unable to find spec file!"
    exit 1
fi
if [ "$VERSION" == "" ]; then
    echo "Unable to determine spec file version!"
    exit 1
fi

# Above was just input parsing and bookkeeping. Here's where things
# actually start to happen.

# Nuke any temporary workspaces so we can start fresh.
rm -rf WORKSPACE/BUILD WORKSPACE/SOURCES WORKSPACE/SPECS WORKSPACE/SRPMS WORKSPACE/RPMS RPMS ||:
# Create a workspace, along with subdirectories SPECS, SOURCES, etc, for
# rpmbuild. Note that it's possible to just pass the tarball to
# rpmbuild directly, but this way makes it easier to run locally and
# browse if something goes wrong.
mkdir -p WORKSPACE/BUILD WORKSPACE/SOURCES WORKSPACE/SPECS WORKSPACE/SRPMS WORKSPACE/RPMS

if [ "$ZYPPER_COMMAND" ]; then
    pushd WORKSPACE/RPMS

    ${ZYPPER_COMMAND} addrepo --no-gpgcheck --check --priority 1 https://arti.hpc.amslabs.hpecorp.net/artifactory/cos-internal-third-party-generic-local/nvidia_driver/sle15_sp5_ncn/${ARCH}/${ARTI_BRANCH} nvidia
    ${ZYPPER_COMMAND} refresh -r nvidia

    ${ZYPPER_COMMAND} addrepo --no-gpgcheck --check --priority 1 /root/repo root-repo
    ${ZYPPER_COMMAND} refresh -r root-repo
	
    popd
fi

excludeList=""
for i in $(find . -name "*.spec"); do
    i=$(basename "${i}")
    if [ "${i}" != "${SPECFILE_BASENAME}" ]; then
        excludeList+="--exclude ${i} "
    fi
done

# See if there are files in the current directory that are sources/patches from the specfile, excluding tarballs.
# If there are, then the assumption here is that this RPM build has sources/patches which need to be copied.
rpmspec --srpm -q --qf "[%{source}\n][%{patch}\n]" ${SPECFILE} | grep -v '[^;]*tar[^;]*'
if [ "$?" -eq "0" ]; then
    # Copy the sources in the specfile to the SOURCES directory via cp (not via tar to avoid leading directories)
    for source in $(rpmspec --srpm -q --qf "[%{source}\n][%{patch}\n]" ${SPECFILE}); do
        if [ -e $source ]; then
           cp $source ./WORKSPACE/SOURCES
       else
           echo "$source doesn't exist, exiting."
           exit 1
       fi
    done

    # Also change the tarball name to a temporary one so we can install the build dependencies without a name conflict.
    SOURCE="tmp-src.tar.gz"
    # Create a list of the sources/patches to exclude in the tar
    SOURCES="$(rpmspec --srpm -q --qf "[--exclude %{source} ][--exclude %{patch} ]" ${SPECFILE})"
    tar --transform "flags=r;s,^,/$RPM_DIR/," --exclude ".git" --exclude "WORKSPACE" $SOURCES ${excludeList} -cvjf "WORKSPACE/SOURCES/$SOURCE" .
else
    tar --transform "flags=r;s,^,/$RPM_DIR/," --exclude ".git" --exclude "$SOURCE" --exclude "WORKSPACE" ${excludeList} -cvjf "WORKSPACE/SOURCES/$SOURCE" .
fi

cp $SPECFILE ./WORKSPACE/SPECS

# Let's see about auto-installing build requirements. On CentOS, it's
# easy, since there's a built-in tool.
if `which yum-builddep &> /dev/null`; then
    yum-builddep --assumeyes ./WORKSPACE/SPECS/$SPECFILE
# On SLES, things are a bit dicier. We set up a fake repo, point zypper
# to it, and tell it to install our RPM (which doesn't exist yet). So
# zypper dutifully gathers up the build requirements of that RPM.
else
    if [ -n "$ZYPPER_COMMAND" ]; then
        # Use the source tarball to create a source RPM. We need to dump
        # the command into a file to avoid Bash messing up the quotes.
        echo "rpmbuild -ts ./WORKSPACE/SOURCES/$SOURCE --target \"$ARCH\" --define \"_os $OS\" --define \"_arch $ARCH\" --define \"_buildname $BUILDNAME\"" > WORKSPACE/deps-helper.sh
        chmod +x WORKSPACE/deps-helper.sh
        ./WORKSPACE/deps-helper.sh
        # Create a dummy repo containing that source RPM.
        createrepo ./WORKSPACE/SRPMS
	createrepo ./WORKSPACE/RPMS
        ${ZYPPER_COMMAND} addrepo -p 1 --no-gpgcheck ./WORKSPACE/SRPMS rpmbuild-local
	${ZYPPER_COMMAND} refresh rpmbuild-local
	${ZYPPER_COMMAND} addrepo -p 1 --no-gpgcheck ./WORKSPACE/RPMS nvidia-local
	${ZYPPER_COMMAND} refresh nvidia-local
        ${ZYPPER_COMMAND} -n source-install --recommends -d $NAME
    else
        echo "WARNING: No package manager found to install dependencies"
    fi
fi


# Now on to the actual build!

# At some point in the above build dependency nonsense, zypper deletes
# our spec file. Thanks zypper!
cp $SPECFILE ./WORKSPACE/SPECS

# ********** BEGIN MODIFICATIONS
# Modifications to original shared-library build script made in order to build
# for multiple kernels in one job

OS_TYPE=`cat /etc/os-release | grep "^ID=" | sed "s/\"//g" | cut -d "=" -f 2`
OS_VERSION=`cat /etc/os-release | grep "^VERSION_ID=" | sed "s/\"//g" | cut -d "=" -f 2`

# This file should have been cloned for is in the runBuildPrep stage.
version_support_file=./hpc-sshot-slingshot-version/shs-kernel-support/${OS_TYPE}${OS_VERSION}.json
if [ -f ${version_support_file} ]; then
    kernels="$(jq -r '.[].kernel | select( . != null)' ${version_support_file})"
else
# If version-support file is missing, build for all installed kernels.
    if [[ ! -v SHS_NEW_BUILD_SYSTEM ]]; then
      kernels="$(for kernel in /usr/src/kernels/*; do basename $kernel; done)"
    else
      case $TARGET_OS in
        rhel*)
          kernels="$(ls -d /usr/src/kernels/*/)"
          ;;
        sle*)
          kernels="$(ls -d /usr/src/linux-obj/${TARGET_ARCH}/*/)"
          ;;
        *)
          echo unknown case
          exit 1
          ;;
      esac
    fi
fi

echo "Building for kernels $kernels"

# Build RPM for every installed kernel version
for kver in $kernels; do
    # Bash wants to stick a bunch of extra quote marks in the middle of the
    # command. To get around that (and to have a record of the command), we
    # dump it into a file.
    echo "--> Building for kernel: $kver"
    echo "rpmbuild -ba ./WORKSPACE/SPECS/$SPECFILE_BASENAME ${GPU_ARGS} --define \"_topdir $TOPDIR\" --define \"kernel_version $kver\" --target \"$ARCH\" --define \"_os $OS\" --define \"_arch $ARCH\" --define \"_buildname $BUILDNAME\" --define \"debug_package $DEBUG_PACKAGE\" " > WORKSPACE/helper.sh
    chmod +x WORKSPACE/helper.sh
    ./WORKSPACE/helper.sh
    RPMRETURN=$?
    if [ $RPMRETURN -ne 0 ]; then
        echo "Failed Build on $kver. Aborting..."
        exit $RPMRETURN
    fi
done

# ********** END MODIFICATIONS

# Move the RPMs and SRPMS into a directory at the root of the repo. This
# should make it easy for Jenkins to grab them.
mkdir RPMS
mv `find WORKSPACE/RPMS | grep rpm$` `find WORKSPACE/SRPMS | grep rpm$` RPMS
chmod a+rwX -R RPMS

# Finish up rpmlint to check for warnings and errors.
rpmlint RPMS/*.rpm

# Return codes from rpmlint:
#  0: OK
#  1: Unspecified error
#  2: Interrupted
# 64: One or more error messages
# 66: Badness level exceeded

# Let's not fail builds for (for example) using static linking.
if [[ $? != 0 && $? != 64 ]]; then
    echo "rpmlint failure!"
    exit 1
fi

exit 0
