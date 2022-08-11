#!/bin/bash
set -ex

ARTI_URL=https://${ARTIFACT_REPO_HOST}/artifactory
OS_TYPE=`cat /etc/os-release | grep "^ID=" | sed "s/\"//g" | cut -d "=" -f 2`
OS_VERSION=`cat /etc/os-release | grep "^VERSION_ID=" | sed "s/\"//g" | cut -d "=" -f 2`

# Override product since we are only using the internal product stream to avoid
# clashing with slingshot10 libfabric
PRODUCT='slingshot-host-software'

echo "$0: --> BRANCH_NAME: '${BRANCH_NAME}'"
echo "$0: --> PRODUCT: '${PRODUCT}'"
echo "$0: --> TARGET_ARCH: '${TARGET_ARCH}'"
echo "$0: --> TARGET_OS: '${TARGET_OS}'"
echo "$0: --> OS_TYPE: '${OS_TYPE}'"
echo "$0: --> OS_VERSION: '${OS_VERSION}'"

if [[ "${BRANCH_NAME}" == release/* ]]; then
    ARTI_LOCATION='rpm-stable-local'
    ARTI_BRANCH=${BRANCH_NAME}
else
    ARTI_LOCATION='rpm-master-local'
    ARTI_BRANCH=dev/master
fi

case "${OBS_TARGET_OS}" in
    cos_2_2*)       COS_BRANCH='release/cos-2.2' ;;
    csm_1_0_11*)    COS_BRANCH='release/cos-2.2' ;;
    cos_2_3*)       COS_BRANCH='release/cos-2.3' ;;
    csm_1_2_0*)     COS_BRANCH='release/cos-2.3' ;;
    cos_2_4*)       COS_BRANCH='release/cos-2.4' ;;
    csm_1_3_0*)     COS_BRANCH='release/cos-2.4' ;;
    *)              COS_BRANCH='dev/master' ;;
esac

echo "$0: --> ARTI_LOCATION: '${ARTI_LOCATION}'"
echo "$0: --> ARTI_BRANCH: '${ARTI_BRANCH}'"
echo "$0: --> COS_BRANCH: '${COS_BRANCH}'"

ZE_ARTI_BRANCH=dev/master

echo "$0: --> ZE_ARTI_BRANCH: '${ZE_ARTI_BRANCH}'"

# Override per OS
with_rocm=0
with_cuda=0
with_ze=0

RPMS="cray-libcxi-devel"

if [[ ${TARGET_OS} == "centos_8" ]]; then
    TARGET_OS="centos_8_ncn"
fi

# ROCM RPM names changed with 4.5.0
# SP2 and release branches still use 4.4
if [[ ${TARGET_OS} == "sle15_sp2_cn" || ${TARGET_OS} == "sle15_sp2_ncn" ]]; then
    ROCR_RPMS="hsa-rocr-dev"
else
    ROCR_RPMS="hsa-rocr-devel"
fi

if [[ ${TARGET_OS} == "sle15_sp3_ncn" && ! ${BRANCH_NAME} == release/* ]]; then
    with_ze=1
    ZE_RPMS="level-zero-devel"
else
    ZE_RPMS=""
fi

if [[ ${TARGET_OS} =~ ^centos || ${TARGET_OS} =~ ^rhel ]]; then
    RPMS+=" libcurl-devel json-c-devel"
else
    RPMS+=" libcurl-devel libjson-c-devel"
fi

if command -v yum > /dev/null; then
    yum-config-manager --add-repo=${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/
    yum-config-manager --setopt=gpgcheck=0 --save
    if [ $OS_TYPE = "rhel"  ] && [ $OS_VERSION = "8.6"  ]; then
        with_rocm=1
        with_cuda=1
        yum-config-manager --add-repo=${ARTI_URL}/radeon-rocm-remote/centos8/5.2/main
        yum-config-manager --add-repo=${ARTI_URL}/radeon-amdgpu-remote/22.10.3/${OS_TYPE}/${OS_VERSION}/main/x86_64/
        yum-config-manager --add-repo=${ARTI_URL}/mirror-nvidia/
        yum-config-manager --add-repo=${ARTI_URL}/pe-internal-rpm-stable-local/nvidia-hpc-sdk/rhel8/
        RPMS+=" rocm-dev hip-devel nvhpc-2022"
    fi

    yum install -y $RPMS
elif command -v zypper > /dev/null; then
    with_cuda=1
    with_rocm=1

    case "${OBS_TARGET_OS}" in
        cos_2_2*)       CUDA_RPMS="nvhpc-2021"
                    ;;
        csm_1_0_11*)    CUDA_RPMS="nvhpc-2021"
                    ;;
        sle15_sp2*)     CUDA_RPMS="nvhpc-2021"
                    ;;
        cos_2_3*)       CUDA_RPMS="nvhpc-2022"
                    ;;
        csm_1_2_0*)     CUDA_RPMS="nvhpc-2022"
                    ;;
        sle15_sp3*)     CUDA_RPMS="nvhpc-2022"
                    ;;
        cos_2_4*)       CUDA_RPMS="nvhpc-2022"
                    ;;
        csm_1_3_0*)     CUDA_RPMS="nvhpc-2022"
                    ;;
        sle15_sp4*)     CUDA_RPMS="nvhpc-2022"
                    ;;
        *)              CUDA_RPMS="nvhpc-2022"
                    ;;
    esac


    if [[ ${OBS_TARGET_OS} == cos* ]]; then
        GDRCOPY_RPMS="gdrcopy-devel"

        case ${COS_BRANCH} in
            release/*)
                COS_ARTI_LOCATION=cos-rpm-stable-local
                ;;
            *)
                COS_ARTI_LOCATION=cos-rpm-master-local
                ;;
        esac

        zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
            --priority 20 --name=cos \
            ${ARTI_URL}/${COS_ARTI_LOCATION}/${COS_BRANCH}/${TARGET_OS} \
            cos
    fi

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 20 --name=${PRODUCT}-${ARTI_LOCATION} \
         ${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${OBS_TARGET_OS}/ \
         ${PRODUCT}-${ARTI_LOCATION}

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 20 --name=cuda \
        ${ARTI_URL}/cos-internal-third-party-generic-local/nvidia_hpc_sdk/${TARGET_OS}/${TARGET_ARCH}/${COS_BRANCH}/ \
        cuda

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 20 --name=rocm \
        ${ARTI_URL}/cos-internal-third-party-generic-local/rocm/latest/${TARGET_OS}/${TARGET_ARCH}/${COS_BRANCH}/ \
        rocm

    if [[ $with_ze -eq 1 ]]; then
        zypper --verbose --non-interactive  addrepo --no-gpgcheck --check \
	--priority 20 --name=ze \
	${ARTI_URL}/cos-internal-third-party-generic-local/intel_gpu/${TARGET_OS}/${TARGET_ARCH}/${ZE_ARTI_BRANCH}/ \
	ze
    fi

    zypper refresh
    zypper --non-interactive --no-gpg-checks install $RPMS $GDRCOPY_RPMS $CUDA_RPMS $ROCR_RPMS $ZE_RPMS
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
