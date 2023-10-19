#!/bin/bash
set -ex

ARTI_URL=https://${ARTIFACT_REPO_HOST}/artifactory
OS_TYPE=`cat /etc/os-release | grep "^ID=" | sed "s/\"//g" | cut -d "=" -f 2`
OS_VERSION=`cat /etc/os-release | grep "^VERSION_ID=" | sed "s/\"//g" | cut -d "=" -f 2`

RHEL_GPU_SUPPORTED_VERSIONS="8.6 8.7 8.8"

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
elif [[ "${CHANGE_TARGET}" == release/* ]]; then
    # CHANGE_TARGET is only set for PR builds and points to the PR target branch
    ARTI_LOCATION='rpm-stable-local'
    ARTI_BRANCH=${CHANGE_TARGET}
else
    ARTI_LOCATION='rpm-master-local'
    ARTI_BRANCH=dev/master
fi

CNE_BRANCH=""

case "${OBS_TARGET_OS}" in
    cos_2_2_*)      COS_BRANCH='release/cos-2.2' ;;
    csm_1_0_11_*)   COS_BRANCH='release/cos-2.2' ;;
    cos_2_3_*)      COS_BRANCH='release/cos-2.3' ;;
    csm_1_2_0_*)    COS_BRANCH='release/cos-2.3' ;;
    cos_2_4_*)      COS_BRANCH='release/cos-2.4' ;;
    csm_1_3_*)      COS_BRANCH='release/cos-2.4' ;;
    sle15_sp4_*)    COS_BRANCH='release/cos-2.5' ;;
    cos_2_5_*)      COS_BRANCH='release/cos-2.5' ;;
    csm_1_4_*)      COS_BRANCH='release/cos-2.5' ;;
    cos_2_6_*)      COS_BRANCH='release/cos-2.6' ;;
    cos_3_0_*)      COS_BRANCH='release/cos-3.0' ;;
    csm_1_5_0_*)    COS_BRANCH='release/cos-3.0' ;;
    sle15_sp5_*)    COS_BRANCH='release/cos-3.0' ;;
    *)              COS_BRANCH='dev/master' ;;
esac

echo "$0: --> ARTI_LOCATION: '${ARTI_LOCATION}'"
echo "$0: --> ARTI_BRANCH: '${ARTI_BRANCH}'"
echo "$0: --> COS_BRANCH: '${COS_BRANCH}'"

# Override per OS
with_rocm=0
with_cuda=0
with_ze=0

RPMS="cray-libcxi-devel kdreg2-devel"

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

if [[ ( ${TARGET_OS} == sle15_sp4* || ${TARGET_OS} == sle15_sp5* ) \
        && ${TARGET_ARCH} == x86_64 ]]; then
    with_ze=1
    ZE_RPMS="level-zero-devel"
else
    ZE_RPMS=""
fi

if [[ ${TARGET_OS} =~ ^centos || ${TARGET_OS} =~ ^rhel ]]; then
    RPMS+=" libcurl-devel json-c-devel cray-libcxi-static "
else
    RPMS+=" libcurl-devel libjson-c-devel cray-libcxi-devel-static "
fi

if command -v yum > /dev/null; then
    yum-config-manager --add-repo=${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/
    yum-config-manager --setopt=gpgcheck=0 --save

    if [ $OS_TYPE = "rhel"  ] && \
            [[ $RHEL_GPU_SUPPORTED_VERSIONS = *$OS_VERSION* ]]; then
        with_rocm=1
        with_cuda=1

        case $OS_VERSION in
        8.6)
            ROCM_VERSION="5.2.3"
            NVIDIA_VERSION="22.7"
            ;;
        8.7)
            ROCM_VERSION="5.5.1"
            NVIDIA_VERSION="23.3"
            ;;
        8.8)
            ROCM_VERSION="5.7"
            NVIDIA_VERSION="23.9"
            ;;
        *)
            echo "GPU software versions not defined for OS version \"${OS_VERSION}\""
            exit 1
        esac

        if [ $OS_VERSION = '8.6' ]; then
            yum-config-manager --add-repo=${ARTI_URL}/radeon-rocm-remote/centos8/${ROCM_VERSION}/main
        else
            yum-config-manager --add-repo=${ARTI_URL}/radeon-rocm-remote/rhel8/${ROCM_VERSION}/main
        fi

        yum-config-manager --add-repo=${ARTI_URL}/mirror-nvhpc/rhel/${TARGET_ARCH}

        RPMS+=" rocm-dev hip-devel nvhpc-${NVIDIA_VERSION} "
    fi

    yum install -y $RPMS
elif command -v zypper > /dev/null; then
    with_cuda=1

    if [[ ${TARGET_ARCH} == x86_64 ]]; then
        with_rocm=1
    fi

    case "${OBS_TARGET_OS}" in
        cos_2_2_*)      CUDA_RPMS="nvhpc-2021"
                    ;;
        csm_1_0_11_*)   CUDA_RPMS="nvhpc-2021"
                    ;;
        sle15_sp2_*)    CUDA_RPMS="nvhpc-2021"
                    ;;
        cos_2_3_*)      CUDA_RPMS="nvhpc-2022"
                    ;;
        csm_1_2_0_*)    CUDA_RPMS="nvhpc-2022"
                    ;;
        sle15_sp3_*)    CUDA_RPMS="nvhpc-2022"
                    ;;
        cos_2_4_*)      CUDA_RPMS="nvhpc-2022"
                    ;;
        csm_1_3_*)      CUDA_RPMS="nvhpc-2022"
                    ;;
        sle15_sp4_*)    CUDA_RPMS="nvhpc-2023"
                    ;;
        cos_2_5_*)      CUDA_RPMS="nvhpc-2023"
                    ;;
        csm_1_4_*)      CUDA_RPMS="nvhpc-2023"
                    ;;
        csm_1_5_*)      CUDA_RPMS="nvhpc"
                    ;;
        cos_2_6_*)      CUDA_RPMS="nvhpc"
                    ;;
        cos_3_0_*)      CUDA_RPMS="nvhpc"
                    ;;
        sle15_sp5_*)    CUDA_RPMS="nvhpc"
                    ;;
        *)              CUDA_RPMS="nvhpc"
                    ;;
    esac


    if [[ ${OBS_TARGET_OS} == cos* ]]; then
        GDRCOPY_RPMS="gdrcopy"
        GDRCOPY_DEVEL="gdrcopy-devel"

        case ${COS_BRANCH} in
            release/cos-3.0)
                COS_ARTI_LOCATION=cne-rpm-stable-local
                CNE_BRANCH='release/cne-1.0'
                ;;
            release/*)
                COS_ARTI_LOCATION=cos-rpm-stable-local
                ;;
            *)
                COS_ARTI_LOCATION=cos-rpm-master-local
                ;;
        esac

        if [ -n "$CNE_BRANCH" ]; then
            zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
            --priority 20 --name=cos \
            ${ARTI_URL}/${COS_ARTI_LOCATION}/${CNE_BRANCH}/${TARGET_OS} \
            cos
        else
            zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
            --priority 20 --name=cos \
            ${ARTI_URL}/${COS_ARTI_LOCATION}/${COS_BRANCH}/${TARGET_OS} \
            cos
        fi 
    fi

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 20 --name=${PRODUCT}-${ARTI_LOCATION} \
         ${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${OBS_TARGET_OS}/ \
         ${PRODUCT}-${ARTI_LOCATION}

    if [ $with_cuda -eq 1 ]; then
        zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
            --priority 20 --name=cuda \
            ${ARTI_URL}/cos-internal-third-party-generic-local/nvidia_hpc_sdk/${TARGET_OS}/${TARGET_ARCH}/${COS_BRANCH}/ \
            cuda

        RPMS+=" ${CUDA_RPMS} "
    fi

    if [ $with_rocm -eq 1 ]; then
        zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
            --priority 20 --name=rocm \
            ${ARTI_URL}/cos-internal-third-party-generic-local/rocm/latest/${TARGET_OS}/${TARGET_ARCH}/${COS_BRANCH}/ \
            rocm

        RPMS+=" ${ROCR_RPMS} "
    fi

    if [[ $with_ze -eq 1 ]]; then
        zypper --verbose --non-interactive  addrepo --no-gpgcheck --check \
            --priority 20 --name=ze \
            ${ARTI_URL}/cos-internal-third-party-generic-local/intel_gpu/latest/${TARGET_OS}/${TARGET_ARCH}/${COS_BRANCH}/ \
            ze

        RPMS+=" ${ZE_RPMS} "
    fi

    zypper refresh

    zypper --non-interactive --no-gpg-checks install $RPMS

    # Force installation of the gdrcopy-devel version that matches the gdrcopy
    # that was installed. Workaround for DST-11466
    if [ -n "${GDRCOPY_DEVEL}" ]; then
        zypper --non-interactive --no-gpg-checks install \
            ${GDRCOPY_RPMS}
        GDRCOPY_VERSION=$(rpm -q gdrcopy --queryformat '%{VERSION}-%{RELEASE}')
        zypper --non-interactive --no-gpg-checks install \
            ${GDRCOPY_DEVEL}-${GDRCOPY_VERSION}
    fi

else
    "Unsupported package manager or package manager not found -- installing nothing"
fi

set -x

if [[ $with_cuda -eq 1 ]]; then
    nvhpc_sdk_versions=($(ls -1 /opt/nvidia/hpc_sdk/Linux_${TARGET_ARCH}/ | sort -rn))
    nvhpc_sdk_version=${nvhpc_sdk_versions[0]}
    nvhpc_cuda_path=/opt/nvidia/hpc_sdk/Linux_${TARGET_ARCH}/$nvhpc_sdk_version/cuda
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
