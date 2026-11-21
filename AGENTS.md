# Agent notes

## Code formatting

After editing any C/C++ source under `src/` (`*.h`, `*.hpp`, `*.cpp`, `*.inl`), run clang-format on the files you changed:

```sh
clang-format -i -style=file <files>
```

- Use **clang-format 18**. It is the version installed in the dev container and the one the
  `clang-format Check` CI workflow (`.github/workflows/clang-format-check.yml`) enforces. Other
  major versions format some constructs differently and will fail CI.
- The style comes from `.clang-format` at the repository root; do not pass inline style options.
- Do not format third-party code. CI skips paths matching `3rdparty/` (the in-tree cxxopts, expected-lite and similar),
  `simd/detail/`, `math/mcadams/` and `tests/vcl.geometry/ref/`. Dependencies come from Conan, so no
  third-party tree lives outside those.
- `SortIncludes` is disabled, so include order is preserved as written.

## Building

Dependencies come from Conan 2 (`conanfile.py`, `conan.lock`, `docs/adr/0001-conan-with-find-package-seam.md`); there are no submodules. Set up the pinned client and remotes, then install and use the generated preset. See the Building section of `README.md` for the full loop and option table.

```sh
python3 -m venv .venv-conan && .venv-conan/bin/pip install -r ci/requirements-conan.txt
ci/conan-setup-remotes.sh
conan install . -pr:h=conan/profiles/linux-gcc -pr:b=default -s build_type=Release --build=missing
cmake --preset conan-release && cmake --build --preset conan-release
```

## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues for `bfierz/vcl`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default roles, except `ready-unattended` / `ready-supervised` replace the `ready-for-*` pair, plus `platform:*` and `backend:*` area labels. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
