#!/bin/bash
set -ex

ARTI_URL=https://arti.dev.cray.com/artifactory

# Override product since we are only using the internal product stream to avoid
# clashing with slingshot10 libfabric
PRODUCT='slingshot-host-software'

echo "$0: --> BRANCH_NAME: '${BRANCH_NAME}'"
echo "$0: --> PRODUCT: '${PRODUCT}'"
echo "$0: --> TARGET_ARCH: '${TARGET_ARCH}'"
echo "$0: --> TARGET_OS: '${TARGET_OS}'"

if [[ "${BRANCH_NAME}" == release/* ]]; then
    ARTI_LOCATION='rpm-stable-local'
    ARTI_BRANCH=${BRANCH_NAME}
else
    ARTI_LOCATION='rpm-master-local'
    ARTI_BRANCH=dev/master
fi

echo "$0: --> ARTI_LOCATION: '${ARTI_LOCATION}'"
echo "$0: --> ARTI_BRANCH: '${ARTI_BRANCH}'"

ZE_ARTI_BRANCH=dev/master

echo "$0: --> ZE_ARTI_BRANCH: '${ZE_ARTI_BRANCH}'"

# Override per OS
with_rocm=0
with_cuda=0
with_ze=0

RPMS="cray-libcxi-devel"
CUDA_RPMS="nvhpc-2021"

if [[ ${TARGET_OS} == "centos_8" ]]; then
    TARGET_OS="centos_8_ncn"
fi

if [[ ${TARGET_OS} == "sle15_sp2_cn" || ${TARGET_OS} == "sle15_sp2_ncn" ]]; then
    ROCR_RPMS="hsa-rocr-dev"
else
    ROCR_RPMS="hsa-rocr-devel"
fi

if [[ ${TARGET_OS} == "sle15_sp3_ncn" ]]; then
    with_ze=1
    ZE_RPMS="level-zero-devel"
else
    ZE_RPMS=""
fi

if [[ ${TARGET_OS} =~ ^centos ]]; then
    RPMS+=" libcurl-devel json-c-devel"
else
    RPMS+=" libcurl-devel libjson-c-devel"
fi

if command -v yum > /dev/null; then
    yum-config-manager --add-repo=${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/
    yum-config-manager --setopt=gpgcheck=0 --save

    yum install -y $RPMS
elif command -v zypper > /dev/null; then
    with_cuda=1
    with_rocm=1

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 20 --name=${PRODUCT}-${ARTI_LOCATION} \
         ${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/ \
         ${PRODUCT}-${ARTI_LOCATION}

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 20 --name=cuda \
        ${ARTI_URL}/cos-internal-third-party-generic-local/nvidia_hpc_sdk/${TARGET_OS}/${TARGET_ARCH}/${ARTI_BRANCH}/ \
        cuda

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 20 --name=rocm \
        ${ARTI_URL}/cos-internal-third-party-generic-local/rocm/latest/${TARGET_OS}/${TARGET_ARCH}/${ARTI_BRANCH}/ \
        rocm

    if [[ $with_ze -eq 1 ]]; then
        zypper --verbose --non-interactive  addrepo --no-gpgcheck --check \
	--priority 20 --name=ze \
	${ARTI_URL}/cos-internal-third-party-generic-local/intel_gpu/${TARGET_OS}/${TARGET_ARCH}/${ZE_ARTI_BRANCH}/ \
	ze
    fi

    zypper refresh
    zypper --non-interactive --no-gpg-checks install $RPMS
    zypper --non-interactive --no-gpg-checks install $CUDA_RPMS
    zypper --non-interactive --no-gpg-checks install $ROCR_RPMS
    if [[ $with_ze -eq 1 ]]; then
        zypper --non-interactive --no-gpg-checks install $ZE_RPMS
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
