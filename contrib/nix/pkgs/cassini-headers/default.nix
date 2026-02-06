{
  lib,
  stdenvNoCC,
  fetchFromGitHub,
}:

stdenvNoCC.mkDerivation rec {
  pname = "cassini-headers";
  version = "13.0.0";

  src = fetchFromGitHub {
    owner = "HewlettPackard";
    repo = "shs-cassini-headers";
    rev = "release/shs-${version}";
    hash = "sha256-Fh8RZFAcqbEcbHCmSFNoIcbHP7uC/CovVdkgAA7SpNQ=";
  };

  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out/include $out/share
    cp -r include/* $out/include/
    # Copy share directory (contains csr_defs.json needed by libcxi build)
    cp -r share/* $out/share/
    runHook postInstall
  '';

  meta = with lib; {
    description = "Hardware definitions and C headers for HPE Cassini/Slingshot network interconnect";
    homepage = "https://github.com/HewlettPackard/shs-cassini-headers";
    license = with licenses; [ gpl2 bsd2 ];
    platforms = platforms.linux;
    maintainers = with maintainers; [ ];
  };
}
