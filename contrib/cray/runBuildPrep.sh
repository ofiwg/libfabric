#!/bin/bash

PROJECT="NETC"
TARGET_ARCH="x86_64"
DEV_NAME="dev"
BRANCH_NAME="master"
IYUM_REPO_NAME_1="os-networking-team"

if [[ "${PRODUCT}" = "" ]]
then
    PRODUCT="shasta-premium"
fi

if [[ "${TARGET_OS}" = "" ]]
then
    TARGET_OS="sle15_cn"
fi

if [[ "${TARGET_OS}" = "centos_8" ]]
then
    TARGET_OS="centos_8_ncn"
fi

echo "$0: --> PRODUCT: '${PRODUCT}'"
echo "$0: --> TARGET_OS: '${TARGET_OS}'"

ZYPPER_OPTS="--verbose --non-interactive"
RPMS="cray-libcxi-devel"
CUDA_RPMS="nvhpc-2021"
ROCR_RPMS="hsa-rocr-dev"

URL_PREFIX="http://car.dev.cray.com/artifactory"
URL_SUFFIX="${TARGET_OS}/${TARGET_ARCH}/${DEV_NAME}/${BRANCH_NAME}"

# URL="http://car.dev.cray.com/artifactory/"
# URL+="${PRODUCT}/${PROJECT}/${TARGET_OS}/${TARGET_ARCH}/"
# URL+="${DEV_NAME}/${BRANCH_NAME}/"
URL="${URL_PREFIX}/${PRODUCT}/${PROJECT}/${URL_SUFFIX}"

URL_HOSTSW="${URL_PREFIX}/slingshot-host-software/${PROJECT}/${URL_SUFFIX}"

URL_INT="${URL_PREFIX}/internal/${PROJECT}/${URL_SUFFIX}"

# URL_SSHOT="http://car.dev.cray.com/artifactory/"
# URL_SSHOT+="${PRODUCT}/SSHOT/${TARGET_OS}/${TARGET_ARCH}/"
# URL_SSHOT+="${DEV_NAME}/${BRANCH_NAME}/"
URL_SSHOT="${URL_PREFIX}/${PRODUCT}/SSHOT/${URL_SUFFIX}"

URL_SSHOT_INT="${URL_PREFIX}/internal/SSHOT/${TARGET_OS}/${TARGET_ARCH}/"
URL_SSHOT_INT+="dev/master/"

CUDA_URL="https://arti.dev.cray.com/artifactory/cos-internal-third-party-generic-local/nvidia_hpc_sdk/${TARGET_OS}/${TARGET_ARCH}/${DEV_NAME}/${BRANCH_NAME}/"

if [[ ${TARGET_OS} != "centos_8_ncn" ]]; then
    with_cuda=1
else
    with_cuda=0
fi


if [[ ${TARGET_OS} == "sle15_sp2_cn" || ${TARGET_OS} == "sle15_sp2_ncn" || ${TARGET_OS} == "sle15_sp3_ncn" || ${TARGET_OS} == "sle15_sp3_cn" ]]; then
    with_rocm=1
else
    with_rocm=0
fi

# No ROCM SP1 support is available.
if [[ $with_rocm -eq 1 ]]; then
    ROCR_URL="https://arti.dev.cray.com/artifactory/cos-internal-third-party-generic-local/rocm/latest/${TARGET_OS}/${TARGET_ARCH}/${DEV_NAME}/${BRANCH_NAME}/"
fi

if command -v yum > /dev/null; then
    yum-config-manager --add-repo=$URL
    yum-config-manager --add-repo=$URL_HOSTSW
    yum-config-manager --add-repo=$URL_SSHOT

    yum-config-manager --setopt=gpgcheck=0 --save

    yum install -y $RPMS
elif command -v zypper > /dev/null; then
    zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 1 \
    	--name=$IYUM_REPO_NAME_1 $URL $IYUM_REPO_NAME_1
    zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 1 \
        --name=$IYUM_REPO_NAME_1 $URL_HOSTSW ${IYUM_REPO_NAME_1}_HOSTSW
    zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 1 \
    	--name=${IYUM_REPO_NAME_1}_SSHOT $URL_SSHOT \
        ${IYUM_REPO_NAME_1}_SSHOT
    zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 1 \
       --name=cuda $CUDA_URL cuda

    zypper refresh
    zypper $ZYPPER_OPTS install $RPMS
    zypper $ZYPPER_OPTS install $CUDA_RPMS

    if [[ $with_rocm -eq 1 ]]; then
        zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 1 \
            --name=rocm $ROCR_URL rocm
        zypper $ZYPPER_OPTS install $ROCR_RPMS
    fi
else
    "Unsupported package manager or package manager not found -- installing nothing"
fi

set -x

if [[ $with_cuda -eq 1 ]]; then
    nvhpc_sdk_versions=($(ls -1 /opt/nvidia/hpc_sdk/Linux_x86_64/ | sort -rn))
    nvhpc_sdk_version=${nvhpc_sdk_versions[0]}
    nvhpc_cuda_path=/opt/nvidia/hpc_sdk/Linux_x86_64/$nvhpc_sdk_version/cuda
    if [[ $nvhpc_sdk_version == "" ]]; then
        echo "CUDA required but not found."
        exit 1
    else
        echo "Using $nvhpc_sdk_version at $nvhpc_cuda_path"

        # Convenient symlink which allows the libfabric build process to not
        # have to call out a specific versioned CUDA directory.
        ln -s $nvhpc_cuda_path /usr/local/cuda

        # The CUDA device driver RPM provides a usable libcuda.so which is
        # required by the libfabric autoconf checks. Since artifactory does not
        # provide this RPM, the cuda-driver-devel-11-0 RPM is installed and
        # provides a stub libcuda.so. But, this stub libcuda.so is installed
        # into a non-lib path. A symlink is created to fix this.
        ln -s /usr/local/cuda/lib64/stubs/libcuda.so \
              /usr/local/cuda/lib64/libcuda.so
    fi
fi

if [[ $with_rocm -eq 1 ]]; then
    rocm_version=$(ls /opt | grep rocm | tr -d "\n")
    if [[ $rocm_version == "" ]]; then
        echo "ROCM required but not found."
        exit 1
    else
        echo "Using ROCM $rocm_version"

        # Convenient symlink which allows the libfabric build process to not
        # have to call out a specific versioned ROCR directory.
        ln -s /opt/$rocm_version /opt/rocm
    fi
fi
