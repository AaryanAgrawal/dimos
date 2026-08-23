{
  description = "Latched e-stop native module for DimOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # Relative git+file: is the only way to reach the local path deps outside this dir (nix#12281).
    dimos-repo = { url = "git+file:../../../.."; flake = false; };
  };

  outputs = { self, nixpkgs, flake-utils, dimos-repo }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        src = pkgs.runCommand "estop-src" {} ''
          mkdir -p $out/dimos/control/estop/rust
          cp -r ${./src} $out/dimos/control/estop/rust/src
          cp ${./Cargo.toml} $out/dimos/control/estop/rust/Cargo.toml
          cp ${./Cargo.lock} $out/dimos/control/estop/rust/Cargo.lock
          # The replay test include_str!s the fixture, and buildRustPackage runs the tests.
          cp ${dimos-repo}/dimos/control/estop/g1_fall_imu.csv $out/dimos/control/estop/

          mkdir -p $out/native/rust
          cp -r ${dimos-repo}/native/rust/dimos-module $out/native/rust/dimos-module
          cp -r ${dimos-repo}/native/rust/dimos-module-macros $out/native/rust/dimos-module-macros
        '';
      in {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "estop";
          version = "0.1.0";

          inherit src;
          cargoRoot = "dimos/control/estop/rust";
          buildAndTestSubdir = "dimos/control/estop/rust";

          cargoHash = "sha256-G2HU2MLWpYU83yTe5Dt3jGFA6pGe4q3PXUdKrqaAQg4=";

          meta.mainProgram = "estop";
        };
      });
}
