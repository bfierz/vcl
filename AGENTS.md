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
- Do not format third-party code. CI skips paths matching `externals/`, `3rdparty/`, `simd/detail/`,
  `math/mcadams/` and `tests/vcl.geometry/ref/`.
- `SortIncludes` is disabled, so include order is preserved as written.
