#!/bin/bash -x

cd

if [ -d hpc-shs-cxi-driver ]; then
    rm -rf hpc-shs-cxi-driver
fi

tar -zxvf cxi-driver-dkms.tar.gz
cd hpc-shs-cxi-driver

. vars.sh
. zypper-local.sh

export ROOT_DIR=`pwd`

APP_MACRO=$(grep -F "%{_app}" ${SPECFILE})
APP_MACRO=$?
if grep -F "%{_app}" ${SPECFILE}; then
    NAME=`echo $SPECFILE | rev | cut -d '.' -f 2- | rev`
    SOURCE=`rpmspec --srpm -q --qf "[%{source}]\n" $NAME.spec | xargs basename`
else
    SOURCE="$(rpmspec --srpm -q --qf "[%{source};]\n" ${SPECFILE}  | grep -o '[^;]*tar[^;]*')"
    NAME="$(rpmspec --srpm -q --qf "[%{name}]\n" ${SPECFILE})"
fi

echo "$0: --> SOURCE: '${SOURCE}'"
echo "$0: --> NAME: '${NAME}'"

repo_remove_add_refresh () {
    # $1 - repo name
    # $2 - repo_url

    ${ZYPPER_COMMAND} lr $1
    res=$?

    if [ $res -eq 0 ]; then
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
    repo_remove_add_refresh "internal-${ARTI_LOCATION}-noos" "${ARTI_URL}/internal-${ARTI_LOCATION}/${ARTI_BRANCH}/noos"

    repo_remove_add_refresh "internal-${ARTI_LOCATION}" "${ARTI_URL}/internal-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}"
    repo_remove_add_refresh "${PRODUCT}-${ARTI_LOCATION}" "${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}"
    repo_remove_add_refresh "${EXTRA}-${ARTI_LOCATION}" "${ARTI_URL}/${EXTRA}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}"
    
    # We need to manually install this RPM (rather than let the build process
    # install it automatically) since it contains the kernel-module-package
    # macros that our specfile uses
    pushd WORKSPACE/RPMS
    ${ZYPPER_COMMAND} install -y kernel-default-devel

    ${ZYPPER_COMMAND} addrepo --no-gpgcheck --check --priority 1 https://arti.hpc.amslabs.hpecorp.net/artifactory/cos-internal-third-party-generic-local/nvidia_driver/${OS}/${ARCH}/${ARTI_BRANCH} nvidia
    ${ZYPPER_COMMAND} refresh -r nvidia

    ${ZYPPER_COMMAND} addrepo --no-gpgcheck --check --priority 1 /root/repo root-repo
    ${ZYPPER_COMMAND} refresh -r root-repo
	
    # This is TEMPORARY until a valid Zypper repos for this stuff are located in Arti.
    wget https://arti.hpc.amslabs.hpecorp.net:443/artifactory/cos-internal-third-party-generic-local/rocm/latest/sle15_sp2_cn/x86_64/release/cos-2.2/rock-dkms-5.11.32.0-999999.el7.noarch.rpm
    wget https://arti.hpc.amslabs.hpecorp.net:443/artifactory/cos-internal-third-party-generic-local/rocm/latest/sle15_sp2_cn/x86_64/release/cos-2.2/rock-dkms-firmware-5.11.32.0-999999.el7.noarch.rpm
    popd

    createrepo ./WORKSPACE/SRPMS
    createrepo ./WORKSPACE/RPMS
    createrepo ./RPMS

    repo_remove_add_refresh "rpmbuild-local" "./WORKSPACE/SRPMS"
    repo_remove_add_refresh "nvidia-local" "./WORKSPACE/RPMS"
    repo_remove_add_refresh "cxi-local" "./RPMS"

    ${ZYPPER_COMMAND} -n source-install -d $NAME

    echo "$0: --> Installing CXI DKMS driver: cray-cxi-driver-dkms=${CXI_VERSION}-SSHOT${PRODUCT_VERSION}_0'"
    ${ZYPPER_COMMAND} in -y cray-cxi-driver-dkms=${CXI_VERSION}-SSHOT${PRODUCT_VERSION}_0
    echo "$0: --> Zypper returned: '$?'"

elif command -v yum > /dev/null; then
    yum-config-manager --setopt=gpgcheck=0 --save
    KERNEL_DEVEL_PKG=kernel-devel

    version_support_file=./hpc-sshot-slingshot-version/shs-kernel-support/${OS_TYPE}${OS_VERSION}.json
    if [ -f ${version_support_file} ]; then
        yum install -y jq
        for repo in $(jq -r '.[].repo | select( . != null)' ${version_support_file}); do
            yum-config-manager --add-repo $repo
        done
        yum -y install $(jq -r '.[].packages[] | select( . != null)' ${version_support_file})
    fi

    if [ $OS_TYPE = "centos" ]; then
        # Pin kernel version to 4.18.0-305 from Centos 8.4
        cat > /etc/yum.repos.d/dst-remote-centos8.repo <<- END
[dst-remote-centos8.4.2105]
name=DST remote centos8.4.2105
baseurl=${ARTI_URL}/mirror-centos8.4/8.4.2105/BaseOS/x86_64/os/
enabled=1
gpgcheck=0
logfile=/var/log/yum.log
exclude=kernel
[dst-remote-centos8.4.2105-appstream]
name=DST remote centos8.4.2105 AppStream
baseurl=${ARTI_URL}/mirror-centos8.4/8.4.2105/AppStream/x86_64/os/
enabled=1
gpgcheck=0
logfile=/var/log/yum.log
exclude=kernel
END
        KERNEL_DEVEL_PKG=kernel-devel-4.18.0-305.25.1.el8_4

        yum clean all
        yum install -y python3-dnf-plugin-versionlock
        yum install -y $KERNEL_DEVEL_PKG
        yum versionlock 'kernel-*'
    elif [ $OS_TYPE = "rhel" ] && [[ $RHEL_GPU_SUPPORTED_VERSIONS = *$OS_VERSION* ]]; then

        case $OS_VERSION in
        8.7)
            AMDGPU_VERSION="5.5.1"
            NVIDIA_VERSION="525.105.17"
            ;;
        8.8)
            AMDGPU_VERSION="5.7"
            NVIDIA_VERSION="535.129.03"
            ;;
        8.9)
            AMDGPU_VERSION="5.7"
            NVIDIA_VERSION="535.129.03"
            ;;
        8.10)
            AMDGPU_VERSION="6.1"
            NVIDIA_VERSION="550.54.15"
            ;;
        9.3)
            AMDGPU_VERSION="5.7"
            NVIDIA_VERSION="535.129.03"
            ;;
        9.4)
            AMDGPU_VERSION="6.1"
            NVIDIA_VERSION="550.54.15"
            ;;
        9.5)
            AMDGPU_VERSION="6.3"
            NVIDIA_VERSION="550.90.07"
            ;;
        *)
            echo "GPU software versions not defined for OS version \"${OS_VERSION}\""
            exit 1
        esac

        ### Set up package repositories for GPU driver sources
        yum-config-manager --add-repo=${ARTI_URL}/radeon-amdgpu-remote/${AMDGPU_VERSION}/${OS_TYPE}/${OS_VERSION}/main/x86_64/
	yum-config-manager --add-repo=${ARTI_URL}/mirror-nvidia/

        # Nvidia repo uses module filtering to select major versions, switch to the appropriate module
        NVIDIA_MAJORVERSION=$(cut -d. -f1 <<< ${NVIDIA_VERSION})
        yum module switch-to -y nvidia-driver:${NVIDIA_MAJORVERSION}

        ### Install GPU driver sources
        yum --nogpgcheck install -y nvidia-kmod-headers-${NVIDIA_VERSION} nvidia-driver-devel-${NVIDIA_VERSION} amdgpu amdgpu-dkms

        NVIDIA_VERSION=$(ls /usr/src | grep nvidia)
        mkdir -p /usr/src/kernel-modules
        ln -s /usr/src/$NVIDIA_VERSION /usr/src/kernel-modules/$NVIDIA_VERSION
    fi

    # This is to get cassini firmware
    yum-config-manager --add-repo=${ARTI_URL}/internal-${ARTI_LOCATION}/${ARTI_BRANCH}/noos/
    yum-config-manager --add-repo=${ARTI_URL}/internal-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/
    yum-config-manager --add-repo=${ARTI_URL}/${PRODUCT}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/
    yum-config-manager --add-repo=${ARTI_URL}/${EXTRA}-${ARTI_LOCATION}/${ARTI_BRANCH}/${TARGET_OS}/

    # We need to manually install this RPM (rather than let the build process
    # install it automatically) since it contains the kernel-module-package
    # macros that our specfile uses
    yum install -y kernel-rpm-macros
else
    "Unsupported package manager or package manager not found -- installing nothing"
fi
