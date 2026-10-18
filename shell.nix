{ pkgs ? import <nixpkgs> {} }:

let
  libPath = pkgs.lib.makeLibraryPath [
    pkgs.zlib
    pkgs.bzip2
    pkgs.xz
    pkgs.stdenv.cc.cc.lib
    pkgs.libgcc
    pkgs.openssl
  ];
in
pkgs.mkShell {
  name = "qqh2025-dev";

  buildInputs = [
    pkgs.python312
    pkgs.uv
    pkgs.git
    pkgs.zlib
    pkgs.bzip2
    pkgs.xz
    pkgs.stdenv.cc.cc.lib
    pkgs.libgcc
    pkgs.openssl
  ];

  shellHook = ''
    # Make Nix libs visible to wheels (numpy, scipy, PyTorch, xgboost, ...)
    export LD_LIBRARY_PATH=${libPath}:$LD_LIBRARY_PATH

    echo "[shell.nix] LD_LIBRARY_PATH set to:"
    echo "  $LD_LIBRARY_PATH"
    echo "[shell.nix] Python: $(python3 --version)"

    if [ ! -d .venv ]; then
      echo "[shell.nix] Creating new virtualenv with uv..."
      uv venv --python $(which python3)
    fi

    source .venv/bin/activate

    echo "[shell.nix] Syncing uv.lock (uv sync --frozen)..."
    uv sync --frozen

    echo "[shell.nix] Environment ready."
  '';
}
