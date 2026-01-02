{
  lib,
  stdenvNoCC,
  fetchFromGitHub,
}:

stdenvNoCC.mkDerivation rec {
  pname = "cxi-driver-headers";
  version = "13.0.0";

  src = fetchFromGitHub {
    owner = "HewlettPackard";
    repo = "shs-cxi-driver";
    rev = "release/shs-${version}";
    hash = "sha256-kJNKuvg0x6gGMPlz1y5EiRReSOqx65dmmdwfqGpVptU=";
  };

  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out/include
    cp -r include/* $out/include/
    runHook postInstall
  '';

  meta = with lib; {
    description = "UAPI headers for HPE CXI driver (Slingshot)";
    homepage = "https://github.com/HewlettPackard/shs-cxi-driver";
    license = licenses.gpl2;
    platforms = platforms.linux;
    maintainers = with maintainers; [ ];
  };
}
