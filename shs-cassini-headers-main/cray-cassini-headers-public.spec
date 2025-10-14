# Copyright 2021,2024 Hewlett Packard Enterprise Development LP
Name:           cray-cassini-headers
Version:        1.1.2
Release:        %(echo ${BUILD_METADATA})
Summary:        Cassini Headers
License:        Dual GPLv2/BSD
Source0:        %{name}-%{version}.tar.gz
Prefix:         /usr

%description
Cassini Headers

%package user
# User package, for external distibution
Summary:    Cassini user headers
BuildArch:  noarch
Provides:   cray-cassini-headers-core = %{version}-%{release}
Obsoletes:  cray-cassini-headers-core < 1.0-SSHOT2.0.0_20220224145456_085d779
%description user
Cassini user headers

%package -n cray-cassini-csr-defs
Summary:    Cassini CSR definitions for PyCXI
BuildArch:  noarch
%description -n cray-cassini-csr-defs
Cassini CSR definitions for PyCXI

%prep
%setup -q -n %{name}-%{version}

%build

%install
rm -rf %{buildroot}

mkdir -p %{buildroot}/%{_includedir}
cp -a %{_builddir}/%{name}-%{version}/include/* %{buildroot}/%{_includedir}/

mkdir -p %{buildroot}/%{_datadir}
cp -a %{_builddir}/%{name}-%{version}/share/* %{buildroot}/%{_datadir}/

%files user
%{_includedir}/cxi_prov_hw.h
%{_includedir}/cassini_user_defs.h
%{_includedir}/cassini_error_defs.h
%{_includedir}/cassini_csr_defaults.h
%{_includedir}/libcassini.h
%{_includedir}/sbl/sbl_csr_common.h
%{_includedir}/sbl/sbl_pml.h
%{_includedir}/sbl/sbl_mb.h
%{_includedir}/cassini_cntr_defs.h
%{_includedir}/cassini_cntr_desc.h
%{_includedir}/cassini-telemetry-items.h
%{_includedir}/cassini-telemetry-ethtool-names.h
%{_includedir}/cassini-telemetry-sysfs-defs.h
%{_includedir}/cassini-telemetry-test.h

%files -n cray-cassini-csr-defs
%{_datadir}/cassini-headers/csr_defs.json

%changelog
