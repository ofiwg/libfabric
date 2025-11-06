#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only
# Copyright 2020 Hewlett Packard Enterprise Development LP

if [[ -v SHS_NEW_BUILD_SYSTEM ]]; then
  . ${CE_INCLUDE_PATH}/buildprep-common.sh

  replace_release_metadata "cray-libcxi.spec"
  install_dependencies "cray-libcxi.spec"
else

set -ex

ARTI_URL=https://arti.hpc.amslabs.hpecorp.net/artifactory
OS_TYPE=`cat /etc/os-release | grep "^ID=" | sed "s/\"//g" | cut -d "=" -f 2`
OS_VERSION=`cat /etc/os-release | grep "^VERSION_ID=" | sed "s/\"//g" | cut -d "=" -f 2`
#PRODUCT=${PRODUCT:-"slingshot-host-software"}

RHEL_GPU_SUPPORTED_VERSIONS="8.10 9.4 9.5"

echo "$0: --> BRANCH_NAME: '${BRANCH_NAME}'"
echo "$0: --> PRODUCT: '${PRODUCT}'"
echo "$0: --> TARGET_ARCH: '${TARGET_ARCH}'"
echo "$0: --> TARGET_OS: '${TARGET_OS}'"
echo "$0: --> OBS_TARGET_OS: '${OBS_TARGET_OS}'"
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

case "${OBS_TARGET_OS}" in
    csm_1_4_*)      GPU_BRANCH='release/cos-2.5' ;;
    cos_3_1_*)      GPU_BRANCH='release/uss-1.1' ;;
    csm_1_5_*)      GPU_BRANCH='release/uss-1.1' ;;
    sle15_sp5_*)    GPU_BRANCH='release/uss-1.1' ;;
    sle15_sp6_*)    GPU_BRANCH='release/uss-1.2' ;;
    cos_3_2_*)      GPU_BRANCH='release/uss-1.2' ;;
    cos_3_3_*)      GPU_BRANCH='release/uss-1.3' ;;
    csm_1_6_*)      GPU_BRANCH='release/uss-1.2' ;;
    *)              GPU_BRANCH='dev/master' ;;
esac

echo "$0: --> ARTI_LOCATION: '${ARTI_LOCATION}'"
echo "$0: --> ARTI_BRANCH: '${ARTI_BRANCH}'"
echo "$0: --> GPU_BRANCH: '${GPU_BRANCH}'"

# Override per OS
with_rocm=0
with_cuda=0

if [[ ( ${TARGET_OS} == sle15_sp4* || ${TARGET_OS} == sle15_sp5* ) \
      && ${TARGET_ARCH} == x86_64 ]]; then
    with_ze=1
    ZE_RPMS="level-zero-devel"
else
    with_ze=0
    ZE_RPMS=""
fi

if command -v yum > /dev/null; then
    yum-config-manager --setopt=gpgcheck=0 --save
    yum-config-manager --add-repo=${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${OBS_TARGET_OS}/

    if [ "$OS_TYPE" = "rhel"  ] && \
            [[ $RHEL_GPU_SUPPORTED_VERSIONS = *$OS_VERSION* ]]; then

        with_cuda=1
        if [ "${TARGET_ARCH}" == x86_64 ]; then
            with_rocm=1
        fi

        case $OS_VERSION in
            8.10)
                ROCM_VERSION="6.1"
                NVIDIA_VERSION="24.3"
                ;;
            9.4)
                ROCM_VERSION="6.1"
                NVIDIA_VERSION="24.3"
                ;;
            9.5)
                ROCM_VERSION="6.3"
                NVIDIA_VERSION="24.11"
                ;;
            *)
                echo "GPU software versions not defined for OS version \"${OS_VERSION}\""
                exit 1
        esac

        if [ $with_rocm -eq 1 ]; then
            if [[ $OS_VERSION =~ ^8\.[0-9]+ ]]; then
                yum-config-manager --add-repo=${ARTI_URL}/radeon-rocm-remote/rhel8/${ROCM_VERSION}/main
            elif [[ $OS_VERSION =~ ^9\.[0-9]+ ]]; then
                yum-config-manager --add-repo=${ARTI_URL}/radeon-rocm-remote/rhel9/${ROCM_VERSION}/main
            else
                echo "Variable: $OS_VERSION does not start with 8 or 9"
                exit 1
            fi
            yum install -y rocm-dev hip-devel
        fi 

        yum-config-manager --add-repo=${ARTI_URL}/mirror-nvhpc/rhel/${TARGET_ARCH}
        yum install -y nvhpc-${NVIDIA_VERSION}

    else
        yum-config-manager --add-repo=${ARTI_URL}/mirror-centos8/Devel/${TARGET_ARCH}/os
    fi

elif command -v zypper > /dev/null; then
    if [[ ${TARGET_ARCH} == x86_64 ]]; then
        with_rocm=1
    fi

    TARGET_OS_SHORT=$(echo $TARGET_OS | sed -e "s/_cn//g" -e "s/_ncn//g")

    with_cuda=1

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 1 --name=${PRODUCT}-${ARTI_LOCATION} \
         ${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${OBS_TARGET_OS}/ \
         ${PRODUCT}-${ARTI_LOCATION}

    if [ $with_cuda -eq 1 ]; then
        if [[ ${GPU_BRANCH} == release/uss-* ]]; then
            zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
                    --priority 1 --name=cuda \
                    ${ARTI_URL}/uss-internal-third-party-rpm-local/nvidia_hpc_sdk/${GPU_BRANCH}/${TARGET_OS_SHORT}/ \
                    cuda
        else
            zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
                    --priority 1 --name=cuda \
                    ${ARTI_URL}/cos-internal-third-party-generic-local/nvidia_hpc_sdk/${TARGET_OS}/${TARGET_ARCH}/${GPU_BRANCH}/ \
                    cuda
        fi
    fi

    if [ $with_rocm -eq 1 ]; then
        if [[ ${GPU_BRANCH} == release/uss-* ]]; then
            zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
                    --priority 1 --name=rocm \
                    ${ARTI_URL}/uss-internal-third-party-rpm-local/rocm/${GPU_BRANCH}/${TARGET_OS_SHORT}/ \
                    rocm
        else
            zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
                    --priority 1 --name=rocm \
                    ${ARTI_URL}/cos-internal-third-party-generic-local/rocm/latest/${TARGET_OS}/${TARGET_ARCH}/${GPU_BRANCH}/ \
                    rocm
        fi
    fi

    if [[ $with_ze -eq 1 ]]; then
        if [[ ${GPU_BRANCH} == release/uss-* ]]; then
            zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
                    --priority 1 --name=ze \
                    ${ARTI_URL}/uss-internal-third-party-rpm-local/intel_gpu/${GPU_BRANCH}/${TARGET_OS_SHORT}/ \
                    ze
        else
            zypper --verbose --non-interactive  addrepo --no-gpgcheck --check \
                    --priority 1 --name=ze \
                ${ARTI_URL}/cos-internal-third-party-generic-local/intel_gpu/latest/${TARGET_OS}/${TARGET_ARCH}/${GPU_BRANCH}/ \
                ze
        fi
    fi

    case "${OBS_TARGET_OS}" in
        csm_1_4_*)      NVIDIA_RPMS="nvhpc-2023"
                        AMD_RPMS="hip-devel"
                    ;;
        csm_1_5_*)      NVIDIA_RPMS="nvhpc"
                        AMD_RPMS="hip-devel"
                    ;;
        csm_1_6_*)      NVIDIA_RPMS="nvhpc"
                        AMD_RPMS="hip-devel"
                    ;;
        sle15_sp5_*)    NVIDIA_RPMS="nvhpc"
                        AMD_RPMS="hip-devel"
                    ;;
        sle15_sp6_*)    NVIDIA_RPMS="nvhpc"
                        AMD_RPMS="hip-devel"
                    ;;
        cos_3_1_*)      NVIDIA_RPMS="nvhpc"
                        AMD_RPMS="hip-devel"
                    ;;
        cos_3_2_*)      NVIDIA_RPMS="nvhpc"
                        AMD_RPMS="hip-devel"
                    ;;
        cos_3_3_*)      NVIDIA_RPMS="nvhpc"
                        AMD_RPMS="hip-devel"
                    ;;
        *)              NVIDIA_RPMS="nvhpc-2023"
                        AMD_RPMS="hip-devel"
                    ;;
    esac

    case ${TARGET_OS} in
        sle15_sp5_*)        SLE_VERSION="15.5"
                    ;;
        sle15_sp6_*)        SLE_VERSION="15.6"
                    ;;
    esac

    zypper --verbose --non-interactive addrepo --no-gpgcheck --check \
        --priority 1 --name=devel:languages:perl \
        ${ARTI_URL}/mirror-opensuse-buildservice/repositories/devel:/languages:/perl/${SLE_VERSION} \
        devel:languages:perl

    zypper --verbose --non-interactive refresh

    RPMS=''
    if [ $with_cuda -eq 1 ]; then
        RPMS+=" ${NVIDIA_RPMS} "
    fi
    if [ $with_rocm -eq 1 ]; then
        RPMS+=" ${AMD_RPMS} "
    fi
    if [ $with_ze -eq 1 ]; then
        RPMS+=" ${ZE_RPMS} "
    fi

    if [ -n "$RPMS" ]; then
        zypper --non-interactive --no-gpg-checks install ${RPMS}
    fi

else
    "Unsupported package manager or package manager not found -- installing nothing"
fi

if [[ $with_cuda -eq 1 ]]; then

    # Specify the directory where you want to search for folders
    search_dir="/opt/nvidia/hpc_sdk/Linux_${TARGET_ARCH}"
    echo "contents of /opt/nvidia/hpc_sdk"
    ls /opt/nvidia/hpc_sdk
    echo
    echo "contents of ${search_dir}"
    ls ${search_dir}

    # Define a pattern to match folders in the "x.y" format
    pattern='^[0-9]+\.[0-9]+$'

    # Initialize variables to keep track of the latest folder and its version
    latest_version=""
    latest_folder=""

    # Iterate through the directories in the search directory
    for dir in "$search_dir"/*; do
        if [[ -d "$dir" && $(basename "$dir") =~ $pattern ]]; then
            version="$(basename "$dir")"
            if [[ -z "$latest_version" || "$version" > "$latest_version" ]]; then
                latest_version="$version"
                latest_folder="$dir"
            fi
        fi
    done

    # Check if any matching folders were found
    if [ -n "$latest_folder" ]; then
        nvhpc_sdk_version="$latest_version"
        echo "Using $nvhpc_sdk_version at $latest_folder"
        nvhpc_cuda_path=$latest_folder/cuda
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

    else
        echo "No matching CUDA folders found."
        exit 1
    fi
fi

if [[ $with_rocm -eq 1 ]]; then

    # Find the ROCm version directory in /opt/
    rocm_version_dir=$(ls -d /opt/rocm-* 2>/dev/null)

    # Check if a ROCm version directory was found
    if [ -n "$rocm_version_dir" ]; then
        # Extract the version from the directory path
        rocm_version=$(basename "$rocm_version_dir")
        update-alternatives --display rocm
        
        # # Check if the version follows the expected format
        # if [[ $rocm_version =~ ^rocm-[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        #     echo "ROCm version: $rocm_version"
        #     ln -s /opt/"$rocm_version" /opt/rocm
        # else
        #     echo "Unexpected directory structure found: $rocm_version"
        #     exit 1
        # fi
    else
        echo "ROCm is not installed in /opt/ or the directory structure is different."
        exit 1
    fi

fi

echo "ROCm Version: ${rocm_version}" > /var/tmp/gpu-versions
echo "Nvidia Version: ${nvhpc_sdk_version}" >> /var/tmp/gpu-versions
echo "GPU Versions File:"
echo "$(</var/tmp/gpu-versions)"

BRANCH=`git branch --show-current || git rev-parse --abbrev-ref HEAD`

if [ -d hpc-shs-version ]; then
    git -C hpc-shs-version pull
else
    if [[ -n "${SHS_LOCAL_BUILD}" ]]; then
        git clone git@github.hpe.com:hpe/hpc-shs-version.git
    else
    	git clone https://$HPE_GITHUB_TOKEN@github.hpe.com/hpe/hpc-shs-version.git
    fi
fi

. hpc-shs-version/scripts/get-shs-version.sh
. hpc-shs-version/scripts/get-shs-label.sh

PRODUCT_VERSION=$(get_shs_version)
PRODUCT_LABEL=$(get_shs_label)

echo "INFO: Using SHS release version from BRANCH: '$BRANCH_NAME'"
echo
echo "INFO: SHS release version '$PRODUCT_VERSION'"

sed -i -e "s/Release:.*/Release: ${PRODUCT_LABEL}${PRODUCT_VERSION}_%(echo \\\${BUILD_METADATA:-1})/g" cray-libcxi.spec
fi
