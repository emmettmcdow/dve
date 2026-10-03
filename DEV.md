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
# Unit tests
zig build test

# Build the Zig example and the XCFramework
./scripts/validate.sh
```

`validate.sh` is deliberately separate from `zig build test`: it is slow, so run it before releases or after touching `bindings/` or `examples/`.
