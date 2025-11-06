#!/bin/bash

# Add OS-appropriate repos for build dependencies

set -ex

if [[ -v SHS_NEW_BUILD_SYSTEM ]]; then  # new build system
  . ${CE_INCLUDE_PATH}/load.sh

  target_branch=$(get_effective_target)
  quality_stream=$(get_artifactory_quality_label)

  add_repository "${ARTI_URL}/asicfwdg-rpm-${quality_stream}-local/${target_branch}/noos" "asicfwdg-${quality_stream}-noos"
  add_repository "${ARTI_URL}/asicfwdg-rpm-${quality_stream}-local/${target_branch}/${TARGET_OS}" "asicfwdg-${quality_stream}"
  add_repository "${ARTI_URL}/${PRODUCT}-rpm-${quality_stream}-local/${target_branch}/${TARGET_OS}" "${PRODUCT}-${quality_stream}"
  
  generate_local_rpmmacros
  install_dependencies "cray-cxi-driver.spec"
else  # START old build system

cat > vars.sh <<- END
ROOT_DIR=`pwd`
PRODUCT=${PRODUCT:-"slingshot"}
EXTRA=${EXTRA:-"slingshot-host-software"}
TARGET_ARCH=${TARGET_ARCH:-"x86_64"}
BRANCH_NAME=${BRANCH_NAME:-"master"}
ARTIFACT_REPO_HOST=${ARTIFACT_REPO_HOST:-"arti.hpc.amslabs.hpecorp.net"}
TARGET_OS=${TARGET_OS:-"sle15_sp5_ncn"}
END

. vars.sh
. zypper-local.sh

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

cat >> vars.sh <<- END
ARTI_URL=${ARTI_URL:-"https://${ARTIFACT_REPO_HOST}/artifactory"}
ARTI_LOCATION=${ARTI_LOCATION:-"rpm-master-local"}
ARTI_BRANCH=${ARTI_BRANCH:-"dev/master"}
OS_TYPE=`cat /etc/os-release | grep "^ID=" | sed "s/\"//g" | cut -d "=" -f 2`
OS_VERSION=`cat /etc/os-release | grep "^VERSION_ID=" | sed "s/\"//g" | cut -d "=" -f 2`
RHEL_GPU_SUPPORTED_VERSIONS="8.10 9.4 9.5"
BRANCH=`git branch --show-current`
END

. vars.sh

OS_MAJOR_VERSION=`echo ${OS_VERSION} | cut -d "." -f 1`

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

echo
echo "INFO: SHS release version '$PRODUCT_VERSION'"

CXI_VERSION=$(grep Version cray-cxi-driver.spec | tr -d " " | cut -d":" -f2)

cat >> vars.sh <<- END
CXI_VERSION=${CXI_VERSION}
PRODUCT_VERSION=${PRODUCT_VERSION}
END

sed -i -e "s/Release:.*/Release: ${PRODUCT_LABEL}${PRODUCT_VERSION}_%(echo \\\${BUILD_METADATA:-1})/g" cray-cxi-driver.spec

echo "$0: --> BRANCH_NAME: '${BRANCH_NAME}'"
echo "$0: --> CHANGE_TARGET: '${CHANGE_TARGET}'"
echo "$0: --> PRODUCT: '${PRODUCT}'"
echo "$0: --> TARGET_ARCH: '${TARGET_ARCH}'"
echo "$0: --> TARGET_OS: '${TARGET_OS}'"
echo "$0: --> OS_VERSION: '${OS_VERSION}'"
echo "$0: --> OS_MAJOR_VERSION: '${OS_MAJOR_VERSION}'"
echo "$0: --> ARTI_LOCATION: '${ARTI_LOCATION}'"
echo "$0: --> ARTI_BRANCH: '${ARTI_BRANCH}'"

if [ -d ${WORKSPACE} ]; then
    rm -rf ${WORKSPACE}
fi

echo "$0: --> WORKSPACE: '${WORKSPACE}'"

repo_remove_add_refresh () {
    # $1 - repo name
    # $2 - repo_url

    if ${ZYPPER_COMMAND} lr $1; then
        echo "Removing repo \"$1\""
        ${ZYPPER_COMMAND} rr $1
    else
    	echo "Repo \"$1\" not present"
    fi
    ${ZYPPER_COMMAND} addrepo --no-gpgcheck --check \
        --priority 1 --name=$1 \
         $2/ \
         $1
    ${ZYPPER_COMMAND} refresh $1
}

if [ "$ZYPPER_COMMAND" ]; then
    mkdir -p $ZYPPER_ROOT/etc/zypp/repos.d
    mkdir -p $ZYPPER_ROOT/var/cache/zypp
    mkdir -p $ZYPPER_ROOT/var/cache/zypp/raw
    mkdir -p $ZYPPER_ROOT/var/cache/zypp/solv
    mkdir -p $ZYPPER_ROOT/var/cache/zypp/packages

    cp -r /etc/zypp ${WORKSPACE}/zypp/etc
    cp -r /var/cache/zypp ${WORKSPACE}/zypp/var/cache

    # This is to get cassini firmware
    repo_remove_add_refresh "asicfwdg-${ARTI_LOCATION}-noos" "${ARTI_URL}/asicfwdg-${ARTI_LOCATION}/${ARTI_BRANCH}/noos"
    repo_remove_add_refresh "asicfwdg-${ARTI_LOCATION}" "${ARTI_URL}/asicfwdg-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}"
    repo_remove_add_refresh "${PRODUCT}-${ARTI_LOCATION}" "${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}"
    repo_remove_add_refresh "${EXTRA}-${ARTI_LOCATION}" "${ARTI_URL}/${EXTRA}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}"

    # We need to manually install this RPM (rather than let the build process
    # install it automatically) since it contains the kernel-module-package
    # macros that our specfile uses
    V1=$(uname -r | cut -d"-" -f1)
    V2=$(uname -r | cut -d"-" -f2)
    
    ${ZYPPER_COMMAND} install -y kernel-default-devel=$V1-$V2.1

    exit 0

elif command -v yum > /dev/null; then
    dnf config-manager --setopt=gpgcheck=0 --save
    KERNEL_DEVEL_PKG=kernel-devel

    version_support_file=./hpc-sshot-slingshot-version/shs-kernel-support/${OS_TYPE}${OS_VERSION}.json
    if [ -f ${version_support_file} ]; then
        yum install -y jq
        for repo in $(jq -r '.[].repo | select( . != null)' ${version_support_file}); do
            dnf config-manager --add-repo $repo
        done
        yum -y install $(jq -r '.[].packages[] | select( . != null)' ${version_support_file})
    fi

    if [ $OS_TYPE = "rhel" ] && [[ $RHEL_GPU_SUPPORTED_VERSIONS = *$OS_VERSION* ]]; then

        case $OS_VERSION in
        8.10)
            AMDGPU_VERSION="6.1"
            NVIDIA_VERSION="550.54.15"
            ;;
        9.4)
            AMDGPU_VERSION="6.1"
            NVIDIA_VERSION="550.54.15"
            ;;
        9.5)
            AMDGPU_VERSION="6.3"
            NVIDIA_VERSION="565.57.01"
            ;;
        *)
            echo "GPU software versions not defined for OS version \"${OS_VERSION}\""
            exit 1
        esac

	if [ ${TARGET_ARCH} = "aarch64" ]; then
	    echo "AMD GPUs not supported for $TARGET_ARCH. Skipping AMD repo add."
        else
	    echo "Adding AMD GPU repo"
	    url="${ARTI_URL}/radeon-amdgpu-remote/${AMDGPU_VERSION}/${OS_TYPE}/${OS_VERSION}/main/${TARGET_ARCH}/"

	    if curl --output /dev/null --silent --head --fail "$url"; then
		    echo "URL $url exists"
	    else
		    echo "URL $url does not exist."

            case $OS_VERSION in
            8.10)
                url="${ARTI_URL}/radeon-amdgpu-remote/${AMDGPU_VERSION}/${OS_TYPE}/8.9/main/${TARGET_ARCH}/"
                ;;
            9.4)
                url="${ARTI_URL}/radeon-amdgpu-remote/${AMDGPU_VERSION}/${OS_TYPE}/9.4/main/${TARGET_ARCH}/"
                ;;
            9.5)
                url="${ARTI_URL}/radeon-amdgpu-remote/${AMDGPU_VERSION}/${OS_TYPE}/9.5/main/${TARGET_ARCH}/"
                ;;
            *)
                echo "GPU software versions not defined for OS version \"${OS_VERSION}\""
                exit 1
            esac
	    fi
        
        ### Set up package repositories for AMD GPU driver sources
        dnf config-manager --add-repo=${url}

	    AMD_GPU_RPMS="amdgpu amdgpu-dkms"
	fi

	if [ ${TARGET_ARCH} = "aarch64" ]; then
	    dnf config-manager --add-repo=${ARTI_URL}/nvidia.com-cuda-remote/${OS_TYPE}${OS_MAJOR_VERSION}/sbsa
        # https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/
	else
	    dnf config-manager --add-repo=${ARTI_URL}/mirror-nvidia/
	fi

	# Nvidia repo uses module filtering to select major versions, switch to the appropriate module
        NVIDIA_MAJORVERSION=$(cut -d. -f1 <<< ${NVIDIA_VERSION})
        dnf module switch-to -y nvidia-driver:${NVIDIA_MAJORVERSION}-dkms
	
        ### Install GPU driver sources
        dnf --nogpgcheck install -y nvidia-kmod-headers-${NVIDIA_VERSION} nvidia-driver-devel-${NVIDIA_VERSION} ${AMD_GPU_RPMS}
        # Check the exit status of the install command
        if [ $? -ne 0 ]; then
            echo "Failed to install GPU driver sources."
            exit 1
        fi

        NVIDIA_VERSION=$(ls /usr/src | grep nvidia)
        mkdir -p /usr/src/kernel-modules
        ln -s /usr/src/$NVIDIA_VERSION /usr/src/kernel-modules/$NVIDIA_VERSION
    fi

    # This is to get cassini firmware
    dnf config-manager --add-repo=${ARTI_URL}/asicfwdg-${ARTI_LOCATION}/${ARTI_BRANCH}/noos/
    dnf config-manager --add-repo=${ARTI_URL}/asicfwdg-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/
    dnf config-manager --add-repo=${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/
    dnf config-manager --add-repo=${ARTI_URL}/${EXTRA}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/

    # We need to manually install this RPM (rather than let the build process
    # install it automatically) since it contains the kernel-module-package
    # macros that our specfile uses
    dnf --nogpgcheck install -y kernel-rpm-macros

    exit 0
else
    "Unsupported package manager or package manager not found -- installing nothing"

    exit 1
fi 
fi  # END: old build system
