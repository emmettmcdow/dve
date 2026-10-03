# Developer Guide

## Release process

Releases involve one artifact:
- **Zig package** — tagged git commit, consumed via `zig fetch`

### 1. Bump the version

Update the version string in both places:
- `build.zig.zon` — `.version = "<version>"`
- `scripts/mk-framework.sh` — `CFBundleShortVersionString` in the generated `Info.plist`

### 2. Validate

```sh
zig build test
./scripts/validate.sh
```

### 3. Commit, tag, and release in one shot

```sh
git add build.zig.zon scripts/mk-framework.sh
git commit -m "Release v<version>"
git push origin main

gh release create v<version> \
  --title "v<version>" \
  --notes "..."
```

`gh release create` tags the current HEAD automatically — this ensures the tag points to the release commit. Do not create the tag separately before this step.

Consumers pick up the release with:

```sh
zig fetch --save git+https://github.com/emmettmcdow/dve#v<version>
```

---

## Release candidates

Use the same process as above but with a `-rc<n>` suffix on the tag (e.g. `v0.0.2-rc1`) and the `--prerelease` flag:

```sh
gh release create v0.0.2-rc1 \
  --title "v0.0.2-rc1" \
  --notes "Release candidate" \
  --prerelease
```

To promote an RC to a final release, run `gh release create` for the final tag. Do not reuse the RC tag — create a fresh one.

---

## Local development

```sh
# Every unit test plus the embedding quality benchmark. Run it on every change.
zig build test

# The benchmark is silent while its scores sit within tolerance of their baselines (see
# "Scoring" in src/benchmark.zig) and fails with the full table otherwise. To see the table
# anyway, or to include the llama model, which is only linked on request:
zig build test -Dbench-report
zig build test -Dllama

# Narrow it by test name, or to one file's tests by its "<file>." prefix
zig build test -Dtest-filter="search mpnet"
zig build test -Dtest-filter="vstore."

# Profiling tests -- not part of `zig build test`
zig build test-profile

# Build the Zig example and the XCFramework
./scripts/validate.sh
```

`validate.sh` is deliberately separate from `zig build test`: it is slow, so run it before releases or after touching `bindings/` or `examples/`.

### Keeping `zig build test` fast

`zig build test` takes a few seconds, and is meant to stay cheap enough to run after every edit. What keeps it that way:

- **`zig build test` builds one test binary per source file and runs them in parallel**, each limited to the tests declared in its own file. A test binary would otherwise also run the tests of everything its root file imports, so most tests ran two or three times. The cost is that a new source file with tests needs its own block in the unit test section of `build.zig`, or its tests will not run.
- **The tests load a precompiled copy of the mpnet model**, which the build compiles once and installs to `zig-out/share/`. The OS reuses its Neural Engine build of a model that stays in one place, and that takes the load from about 3s per test binary to under 0.1s. The compile needs `coremlcompiler` from Xcode; pass `-Dprecompile-model=false` to do without it. The first run after the model is installed or reinstalled is slow, around 10s, while the OS builds its copy.
- **`pfile.syncFd` skips `F_FULLFSYNC` in test builds** and uses plain `fsync`. No test can tell them apart, and the full flush was a third of the run time. One test in `pfile.zig` still makes the call directly.
- **Do not wait with a fixed sleep.** Poll for the condition with a deadline, as the `embedTextAsync` test does.
