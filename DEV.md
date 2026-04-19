# Developer Guide

## Release process

Releases involve two artifacts:
- **Zig package** — tagged git commit, consumed via `zig fetch`
- **Swift XCFramework** — `DVECore-<version>.xcframework.zip` uploaded to GitHub Releases, consumed via SPM `binaryTarget`

### 1. Build the XCFramework

From the repo root:
```sh
zig build xcframework
(cd zig-out && zip -r DVECore-<version>.xcframework.zip DVECore.xcframework)
swift package compute-checksum zig-out/DVECore-<version>.xcframework.zip
```

### 2. Update Package.swift

Update `bindings/swift/Package.swift` with the final release URL and checksum **before** creating the tag. The URL is deterministic:

```
https://github.com/emmettmcdow/dve/releases/download/<tag>/DVECore-<version>.xcframework.zip
```

```swift
.binaryTarget(
    name: "DVECore",
    url: "https://github.com/emmettmcdow/dve/releases/download/v0.0.2/DVECore-0.0.2.xcframework.zip",
    checksum: "<output of swift package compute-checksum>"
),
```

### 3. Commit, tag, and release in one shot

```sh
git add bindings/swift/Package.swift
git commit -m "Release v0.0.2"
git push origin main

gh release create v0.0.2 zig-out/DVECore-<version>.xcframework.zip \
  --title "v0.0.2" \
  --notes "..."
```

`gh release create` tags the current HEAD automatically — this ensures the tag points to the commit that contains the correct `Package.swift`. Do not create the tag separately before this step.

### 4. Validate

```sh
./scripts/validate.sh
```

---

## Release candidates

Use the same process as above but with a `-rc<n>` suffix on the tag (e.g. `v0.0.2-rc1`) and `--prerelease` flag:

```sh
gh release create v0.0.2-rc1 DVECore-0.0.2.xcframework.zip \
  --title "v0.0.2-rc1" \
  --notes "Release candidate" \
  --prerelease
```

To promote an RC to a final release, build a new zip (even if the XCFramework hasn't changed), update `Package.swift` with the final tag URL, commit, and run `gh release create` for the final tag. Do not reuse the RC tag — create a fresh one.

> **Note:** GitHub draft releases return 404 to unauthenticated requests. SPM cannot download from them. Always use a published release (including pre-releases) when testing SPM integration.

---

## Local development

During development, `bindings/swift/Package.swift` should point to the locally built XCFramework:

```swift
.binaryTarget(
    name: "DVECore",
    path: "../../zig-out/DVECore.xcframework"
),
```

Switch back to this form after a release so that `./scripts/validate.sh` works without a live GitHub release. The remote `binaryTarget(url:checksum:)` form is only committed at the moment of tagging.

```sh
# Build xcframework, then validate all bindings and examples
zig build xcframework
./scripts/validate.sh
```
