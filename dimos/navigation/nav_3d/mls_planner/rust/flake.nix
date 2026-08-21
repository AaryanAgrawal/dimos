{
  description = "MLS planner native module for DimOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # Relative git+file: will be deprecated (nix#12281) but there's no
    # viable alternative for reaching local path deps outside the flake dir currently
    # presumably an alternative will be added before this is removed.
    dimos-repo = { url = "git+file:../../../../.."; flake = false; };
  };

  outputs = { self, nixpkgs, flake-utils, dimos-repo }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        src = pkgs.runCommand "mls-planner-src" {} ''
          mkdir -p $out/dimos/navigation/nav_3d/mls_planner/rust
          cp -r ${./src} $out/dimos/navigation/nav_3d/mls_planner/rust/src
          cp ${./Cargo.toml} $out/dimos/navigation/nav_3d/mls_planner/rust/Cargo.toml
          cp ${./Cargo.lock} $out/dimos/navigation/nav_3d/mls_planner/rust/Cargo.lock

          mkdir -p $out/native/rust
          cp -r ${dimos-repo}/native/rust/dimos-module $out/native/rust/dimos-module
          cp -r ${dimos-repo}/native/rust/dimos-module-macros $out/native/rust/dimos-module-macros
        '';
      in {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "mls-planner";
          version = "0.1.0";

          inherit src;
          cargoRoot = "dimos/navigation/nav_3d/mls_planner/rust";
          buildAndTestSubdir = "dimos/navigation/nav_3d/mls_planner/rust";

          cargoHash = "sha256-7Yme4uQ9iFOsRbmHQF3cfsGfsFqigKnSHb7of9u2B7E=";

          meta.mainProgram = "mls_planner";
        };
      });
}
