{
  description = "A fully local MCP server for RAG over PDFs, DOCX, and plaintext files";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      flake-parts,
      nixpkgs,
      git-hooks,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      perSystem =
        {
          pkgs,
          lib,
          system,
          ...
        }:
        let
          # Runtime dependencies that need to be available in PATH
          runtimeDeps = with pkgs; [
            tesseract
          ];

          # Load the uv workspace from the current directory
          workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

          # Create package overlay from workspace
          overlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          # Construct the Python package set
          python = pkgs.python313;

          pythonSet =
            (pkgs.callPackage pyproject-nix.build.packages {
              inherit python;
            }).overrideScope
              (
                lib.composeManyExtensions [
                  pyproject-build-systems.overlays.default
                  overlay
                ]
              );

          # The virtual environment with all dependencies
          venv = pythonSet.mkVirtualEnv "mcp-local-rag-env" workspace.deps.default;

          # Wrapped package that includes system dependencies in PATH
          mcp-local-rag = pkgs.symlinkJoin {
            name = "mcp-local-rag";
            paths = [ venv ];
            buildInputs = [ pkgs.makeWrapper ];
            postBuild = ''
              # Wrap the main executable to include runtime deps in PATH
              wrapProgram $out/bin/mcp-local-rag \
                --prefix PATH : ${lib.makeBinPath runtimeDeps} \
                --prefix LD_LIBRARY_PATH : ${
                  lib.makeLibraryPath [
                    pkgs.stdenv.cc.cc.lib
                    pkgs.zlib
                  ]
                }
            '';
          };

          # Git hooks configuration
          pre-commit-check = git-hooks.lib.${system}.run {
            src = ./.;
            hooks = {
              # General
              check-added-large-files.enable = true;
              check-case-conflicts.enable = true;
              check-executables-have-shebangs.enable = true;
              check-merge-conflicts.enable = true;
              check-shebang-scripts-are-executable.enable = true;
              check-symlinks.enable = true;
              check-toml.enable = true;
              check-yaml.enable = true;
              detect-private-keys.enable = true;
              end-of-file-fixer.enable = true;
              fix-byte-order-marker.enable = true;
              forbid-new-submodules.enable = true;
              mixed-line-endings.enable = true;
              trim-trailing-whitespace.enable = true;

              # Python
              check-python.enable = true;
              check-builtin-literals.enable = true;
              check-docstring-first.enable = true;
              python-debug-statements.enable = true;
              ruff.enable = true;
              ruff-format.enable = true;

              # Nix
              nixfmt.enable = true;
            };
          };
        in
        {
          # Export the wrapped package
          packages = {
            default = mcp-local-rag;
            inherit mcp-local-rag venv;
          };

          # Pre-commit checks
          checks = {
            inherit pre-commit-check;
          };

          # Development shell
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              nodejs_24
              tesseract
              uv
            ];

            env = {
              UV_NO_MANAGED_PYTHON = 1;
              UV_PYTHON_DOWNLOADS = "never";
              # Required for numpy/torch native libraries on NixOS
              LD_LIBRARY_PATH = lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ];
            };

            nativeBuildInputs = with pkgs; [
              python313
            ];

            inherit (pre-commit-check) shellHook;
            buildInputs = pre-commit-check.enabledPackages;
          };
        };

      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
    };
}
