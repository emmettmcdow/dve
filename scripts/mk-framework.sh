#!/bin/sh
# Assembles DVECore.framework from a universal dylib, model files, and headers.
# Usage: mk-framework.sh <dylib> <mlpackage_dir> <tokenizer_json> <output_framework_dir>
set -e

DYLIB="$1"
MODEL_PACKAGE="$2"
TOKENIZER="$3"
FRAMEWORK_DIR="$4"

rm -rf "$FRAMEWORK_DIR"
mkdir -p "$FRAMEWORK_DIR/Headers"
mkdir -p "$FRAMEWORK_DIR/Modules"
mkdir -p "$FRAMEWORK_DIR/Resources"

# Copy dylib into framework. Install name is set at link time via -install_name.
cp "$DYLIB" "$FRAMEWORK_DIR/DVECore"

# Public header
cp bindings/c/include/dve.h "$FRAMEWORK_DIR/Headers/"

# Framework-style module map (required for Clang module import)
cat > "$FRAMEWORK_DIR/Modules/module.modulemap" <<'EOF'
framework module DVECore {
    umbrella header "dve.h"
    export *
    module * { export * }
}
EOF

# Info.plist — sets the bundle identifier callers use to locate the model resources.
cat > "$FRAMEWORK_DIR/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>DVECore</string>
    <key>CFBundleIdentifier</key>
    <string>com.emmettmcdow.DVECore</string>
    <key>CFBundleName</key>
    <string>DVECore</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.3</string>
    <key>CFBundleVersion</key>
    <string>1</string>
</dict>
</plist>
EOF

# Model resources. dve does not resolve these itself; callers find them via the bundle
# identifier above and pass their paths to dve_init.
cp -r "$MODEL_PACKAGE" "$FRAMEWORK_DIR/Resources/"
cp "$TOKENIZER" "$FRAMEWORK_DIR/Resources/"
