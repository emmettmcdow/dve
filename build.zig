pub fn build(b: *std.Build) !void {
    const debug = b.option(bool, "debug-output", "Show debug output") orelse false;
    const test_filter: ?[]const u8 = b.option(
        []const u8,
        "test-filter",
        "Filter to select specific tests",
    );
    const use_lldb = b.option(bool, "lldb", "Run tests under lldb debugger") orelse false;
    const quant = b.option(
        StorageQuantize,
        "storage-quantize",
        "Whether to quantize vectors before storage (none, f_16 or i_8)",
    ) orelse .none;
    // Which embedding model to use is a call-site choice, but llama.cpp is an
    // external dependency: it has to be linked or not at build time. Off by
    // default, so dve still builds with no llama.cpp present.
    const llama = b.option(
        bool,
        "llama",
        "Link the llama.cpp embedding backend (default: false)",
    ) orelse false;
    // Root of a built llama.cpp checkout: needs include/, ggml/include/ and the
    // shared libraries in build/bin/. Only consulted when -Dllama is set.
    const llama_root = b.option(
        []const u8,
        "llama-path",
        "Path to a built llama.cpp checkout (default: $HOME/llama.cpp)",
    ) orelse b.pathJoin(&.{ std.posix.getenv("HOME") orelse ".", "llama.cpp" });
    // Overrides the gguf baked into src/llama_bridge.c. The DVE_LLAMA_MODEL
    // environment variable overrides both at run time.
    const llama_model_path = b.option(
        []const u8,
        "llama-model",
        "Path to the .gguf embedding model (default: <llama-path>/build/bin/nomic-embed-text-v1.5.f32.gguf)",
    ) orelse b.pathJoin(&.{ llama_root, "build", "bin", "nomic-embed-text-v1.5.f32.gguf" });

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    ///////////////////////
    // Mpnet Model Fetch //
    ///////////////////////
    const coreml_models = b.dependency("coreml_models", .{});
    const mpnet_model_path = coreml_models.path("all_mpnet_base_v2/all_mpnet_base_v2.mlpackage");
    const mpnet_tokenizer_path = coreml_models.path("all_mpnet_base_v2/tokenizer.json");

    // Install the mpnet runtime assets (model + tokenizer) to zig-out/share/ so the
    // exe can find them at their default relative paths.
    // This covers dve's own builds only -- a dependency's install steps write to
    // the dependency's private prefix, not the consumer's. Consumers call
    // `installModels` below instead.
    for (addModelInstalls(b, coreml_models)) |s| b.getInstallStep().dependOn(s);

    ////////////////////
    // Dependencies   //
    ////////////////////
    const objc_dep = b.dependency("zig_objc", .{
        .target = target,
        .optimize = optimize,
    });
    const tracy_enable = optimize == .Debug;
    const tracy_dep = b.dependency("tracy", .{
        .target = target,
        .optimize = optimize,
        .tracy_enable = tracy_enable,
        .tracy_callstack = @as(u32, 62),
        // On-demand: collect nothing until a `tracy` server connects, so the
        // client doesn't buffer profiling data unbounded when unattached. Must
        // match nana's tracy options exactly (see nana build.zig addTracy).
        .tracy_on_demand = true,
    });

    ////////////////////
    // Config modules //
    ////////////////////
    const real_options = b.addOptions();
    real_options.addOption(bool, "debug", debug);
    real_options.addOption(StorageQuantize, "quant", quant);
    real_options.addOption(bool, "llama", llama);

    // Fake config used for storage/util tests that don't need real embeddings.
    const fake_options = b.addOptions();
    fake_options.addOption(bool, "debug", debug);
    fake_options.addOption(StorageQuantize, "quant", .none);
    fake_options.addOption(bool, "llama", false);

    ////////////////////
    // Public Module  //
    ////////////////////
    const dve_mod = b.addModule("dve", .{
        .root_source_file = b.path("src/root.zig"),
        .imports = &.{
            .{ .name = "config", .module = real_options.createModule() },
            .{ .name = "objc", .module = objc_dep.module("objc") },
            .{ .name = "tracy", .module = tracy_dep.module("tracy") },
        },
    });
    // Carried on the module itself so consumers don't link these by hand: a
    // Compile step inherits frameworks from every module in its import graph.
    dve_mod.linkFramework("NaturalLanguage", .{});
    dve_mod.linkFramework("CoreML", .{});
    dve_mod.linkFramework("Foundation", .{});
    // Same idea for the llama bridge: carried on the module, so a consumer that
    // builds with -Dllama gets the bridge and the libllama link for free.
    const llama_bridge: ?LlamaBridge = if (llama)
        llamaBridgeLib(b, target, optimize, llama_root, llama_model_path)
    else
        null;
    if (llama_bridge) |bridge| bridge.link(dve_mod);
    // vstore.zig does its file IO through pfile.zig, a thin shim onto libc.
    dve_mod.link_libc = true;
    ////////////////
    // Unit Tests //
    ////////////////
    const filters: []const []const u8 = if (test_filter) |f| &.{f} else &.{};

    const runTest = struct {
        fn run(builder: *std.Build, artifact: *std.Build.Step.Compile, lldb: bool) *RunStep {
            if (lldb) {
                const r = RunStep.create(builder, "lldb test");
                r.addArgs(&.{ "lldb", "--" });
                r.addArtifactArg(artifact);
                return r;
            }
            return builder.addRunArtifact(artifact);
        }
    }.run;

    // Helper to wire up ObjC + tracy imports and framework/lib links for a test.
    const addDeps = struct {
        fn real(
            t: *std.Build.Step.Compile,
            cfg: *std.Build.Step.Options,
            objc: *std.Build.Dependency,
            tr: *std.Build.Dependency,
            tr_enable: bool,
        ) void {
            t.root_module.addOptions("config", cfg);
            t.root_module.addImport("objc", objc.module("objc"));
            t.root_module.addImport("tracy", tr.module("tracy"));
            t.root_module.linkFramework("NaturalLanguage", .{});
            t.root_module.linkFramework("CoreML", .{});
            t.root_module.linkFramework("Foundation", .{});
            // vector.zig -> vstore.zig -> pfile.zig, which is a thin shim onto libc.
            t.root_module.link_libc = true;
            if (tr_enable) {
                t.root_module.linkLibrary(tr.artifact("tracy"));
                t.root_module.link_libcpp = true;
            }
        }
    }.real;

    // vec_storage and note_id_map tests use fake config + tracy only (no ObjC).
    const test_vec_storage = b.step("test-vec_storage", "run tests for src/vec_storage.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/vec_storage.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        t.root_module.addOptions("config", fake_options);
        t.root_module.addImport("tracy", tracy_dep.module("tracy"));
        if (tracy_enable) {
            t.root_module.linkLibrary(tracy_dep.artifact("tracy"));
            t.root_module.link_libcpp = true;
        }
        test_vec_storage.dependOn(&runTest(b, t, use_lldb).step);
    }

    const test_pfile = b.step("test-pfile", "run tests for src/pfile.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/pfile.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        // pfile is a thin shim straight onto libc -- open, pread, pwrite, fsync, fcntl --
        // so it needs libc and nothing else. No config, no tracy.
        t.root_module.link_libc = true;
        test_pfile.dependOn(&runTest(b, t, use_lldb).step);
    }

    const test_vstore = b.step("test-vstore", "run tests for src/vstore.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/vstore.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        // vstore does its file IO through src/pfile.zig, straight onto libc.
        t.root_module.link_libc = true;
        t.root_module.addOptions("config", fake_options);
        t.root_module.addImport("tracy", tracy_dep.module("tracy"));
        if (tracy_enable) {
            t.root_module.linkLibrary(tracy_dep.artifact("tracy"));
            t.root_module.link_libcpp = true;
        }
        test_vstore.dependOn(&runTest(b, t, use_lldb).step);
    }

    const test_codes = b.step("test-codes", "run tests for src/codes.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/codes.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        // codes.zig knows nothing about storage, embeddings, or config -- but it saves and
        // loads itself through pfile.zig, which is a thin shim onto libc.
        t.root_module.link_libc = true;
        test_codes.dependOn(&runTest(b, t, use_lldb).step);
    }

    const test_note_id_map = b.step("test-note_id_map", "run tests for src/note_id_map.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/note_id_map.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        t.root_module.addOptions("config", fake_options);
        t.root_module.addImport("tracy", tracy_dep.module("tracy"));
        if (tracy_enable) {
            t.root_module.linkLibrary(tracy_dep.artifact("tracy"));
            t.root_module.link_libcpp = true;
        }
        test_note_id_map.dependOn(&runTest(b, t, use_lldb).step);
    }

    const test_util = b.step("test-util", "run tests for src/util.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/util.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        // util.zig has no external deps beyond std
        test_util.dependOn(&runTest(b, t, use_lldb).step);
    }

    const test_tokenizer = b.step("test-tokenizer", "run tests for src/tokenizer.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/tokenizer.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        test_tokenizer.dependOn(&runTest(b, t, use_lldb).step);
    }

    const test_embed = b.step("test-embed", "run tests for src/embed.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/embed.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        addDeps(t, real_options, objc_dep, tracy_dep, tracy_enable);
        // Compiles src/ directly, so it does not inherit the bridge from dve_mod.
        if (llama_bridge) |bridge| bridge.link(t.root_module);
        const run = runTest(b, t, use_lldb);
        for (addModelInstalls(b, coreml_models)) |s| run.step.dependOn(s);
        test_embed.dependOn(&run.step);
    }

    const test_vector = b.step("test-vector", "run tests for src/vector.zig");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/vector.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        addDeps(t, real_options, objc_dep, tracy_dep, tracy_enable);
        // Compiles src/ directly, so it does not inherit the bridge from dve_mod.
        if (llama_bridge) |bridge| bridge.link(t.root_module);
        const run = runTest(b, t, use_lldb);
        for (addModelInstalls(b, coreml_models)) |s| run.step.dependOn(s);
        test_vector.dependOn(&run.step);
    }

    const test_benchmark = b.step("test-benchmark", "run embedding quality benchmark tests");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/benchmark.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        t.root_module.addImport("dve", dve_mod);
        addDeps(t, real_options, objc_dep, tracy_dep, tracy_enable);
        const run = runTest(b, t, use_lldb);
        for (addModelInstalls(b, coreml_models)) |s| run.step.dependOn(s);
        test_benchmark.dependOn(&run.step);
    }

    const test_profile = b.step("test-profile", "run profiling tests (not included in test step)");
    {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/profile.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .filters = if (test_filter != null) filters else &.{},
        });
        t.root_module.addImport("dve", dve_mod);
        addDeps(t, real_options, objc_dep, tracy_dep, tracy_enable);
        const run = runTest(b, t, use_lldb);
        for (addModelInstalls(b, coreml_models)) |s| run.step.dependOn(s);
        test_profile.dependOn(&run.step);
    }

    ///////////////////
    // Wikitest      //
    ///////////////////
    // Scaling harness that ingests and searches the Simple Wikipedia corpus.
    // Builds only -- invoke the binary yourself, from wikitest/ so that its
    // default corpus and database paths resolve next to download.sh:
    //   zig build wikitest -Doptimize=ReleaseFast
    //   (cd wikitest && ../zig-out/bin/wikitest embed --limit 1000)
    const wikitest_step = b.step("wikitest", "Build the Wikipedia scaling harness");
    {
        const exe = b.addExecutable(.{
            .name = "wikitest",
            .root_module = b.createModule(.{
                .root_source_file = b.path("wikitest/main.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("dve", dve_mod);
        addDeps(exe, real_options, objc_dep, tracy_dep, tracy_enable);

        // Installed by this step only, so a plain `zig build` doesn't build the
        // harness. The embedder resolves model assets relative to the
        // executable, so the shared install step has to run too.
        wikitest_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        wikitest_step.dependOn(b.getInstallStep());
    }

    ///////////////////
    // Storebench    //
    ///////////////////
    // A/B harness: vec_storage.zig (v1, RAM-resident) against vstore.zig (v2, disk-resident).
    // Builds only -- invoke the binary yourself:
    //   zig build storebench -Doptimize=ReleaseFast
    //   ./zig-out/bin/storebench --n 100000
    const storebench_step = b.step("storebench", "Build the v1-vs-v2 storage benchmark");
    {
        const exe = b.addExecutable(.{
            .name = "storebench",
            .root_module = b.createModule(.{
                .root_source_file = b.path("experiments/storebench/main.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("dve", dve_mod);
        addDeps(exe, real_options, objc_dep, tracy_dep, tracy_enable);
        storebench_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    ///////////////////
    // Binrecall     //
    ///////////////////
    // Does a 1-bit code surface what exact cosine says is the answer? Reads a corpus of real
    // embeddings out of an existing database; builds nothing and writes nothing.
    //   zig build binrecall -Doptimize=ReleaseFast
    //   ./zig-out/bin/binrecall --queries 300
    const binrecall_step = b.step("binrecall", "Build the 1-bit recall experiment");
    {
        const exe = b.addExecutable(.{
            .name = "binrecall",
            .root_module = b.createModule(.{
                .root_source_file = b.path("experiments/binrecall/main.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("dve", dve_mod);
        addDeps(exe, real_options, objc_dep, tracy_dep, tracy_enable);
        binrecall_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    ///////////////////
    // Hamscan       //
    ///////////////////
    // How fast a 1-bit codes array can be scanned in RAM, swept from cache-resident to well
    // past it. Pure std, no dve import -- it measures a pattern, not a format.
    //   zig build hamscan -Doptimize=ReleaseFast
    //   ./zig-out/bin/hamscan
    const hamscan_step = b.step("hamscan", "Build the Hamming scan throughput benchmark");
    {
        const exe = b.addExecutable(.{
            .name = "hamscan",
            .root_module = b.createModule(.{
                .root_source_file = b.path("experiments/hamscan/main.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        hamscan_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    ///////////////////
    // Codesbench    //
    ///////////////////
    // Validates src/codes.zig against the corpus binrecall measured, so a wrong encoder
    // cannot hide behind self-consistent unit tests.
    //   zig build codesbench -Doptimize=ReleaseFast
    //   ./zig-out/bin/codesbench
    const codesbench_step = b.step("codesbench", "Build the codes.zig validation harness");
    {
        const exe = b.addExecutable(.{
            .name = "codesbench",
            .root_module = b.createModule(.{
                .root_source_file = b.path("experiments/codesbench/main.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("dve", dve_mod);
        addDeps(exe, real_options, objc_dep, tracy_dep, tracy_enable);
        codesbench_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(test_vec_storage);
    test_step.dependOn(test_pfile);
    test_step.dependOn(test_vstore);
    test_step.dependOn(test_codes);
    test_step.dependOn(test_note_id_map);
    test_step.dependOn(test_util);
    test_step.dependOn(test_tokenizer);
    test_step.dependOn(test_embed);
    test_step.dependOn(test_vector);
    test_step.dependOn(test_benchmark);

    ///////////////////
    // XCFramework   //
    ///////////////////
    // Builds DVECore.xcframework for use by Swift/C consumers.
    // Both models are compiled in; the C bindings pick one at runtime (bindings/c/src/intf.zig).
    // The mpnet model files are bundled in Resources/ so that model works out of the box.
    const xcfw_step = b.step("xcframework", "Build DVECore.xcframework");
    {
        const arm_target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos });
        const x86_target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .macos });
        const xcfw_optimize: std.builtin.OptimizeMode = .ReleaseFast;

        const xcfw_options = b.addOptions();
        xcfw_options.addOption(bool, "debug", false);
        // Pinned like the other options above: the shipped framework's storage format does
        // not follow -Dstorage-quantize.
        xcfw_options.addOption(StorageQuantize, "quant", StorageQuantize.none);
        // The shipped framework never carries the llama.cpp backend: it would
        // drag libllama.dylib and the libggml-*.dylib set into the bundle.
        xcfw_options.addOption(bool, "llama", false);

        // Tracy must always be disabled in the xcframework. When tracy_enable=true
        // Tracy starts C++ background threads (via global constructors) that
        // interfere with Apple's NLEmbedding initialization on the main thread.
        const xcfw_tracy_arm = b.dependency("tracy", .{
            .target = arm_target,
            .optimize = xcfw_optimize,
            .tracy_enable = false,
        });
        const xcfw_tracy_x86 = b.dependency("tracy", .{
            .target = x86_target,
            .optimize = xcfw_optimize,
            .tracy_enable = false,
        });

        const xcfw_targets = [2]std.Build.ResolvedTarget{ arm_target, x86_target };
        var libs: [2]std.Build.LazyPath = undefined;

        for (xcfw_targets, 0..) |xcfw_target, i| {
            const xcfw_tracy = if (i == 0) xcfw_tracy_arm else xcfw_tracy_x86;
            const lib = b.addLibrary(.{
                .linkage = .dynamic,
                .name = "dve",
                .root_module = b.createModule(.{
                    .root_source_file = b.path("bindings/c/src/intf.zig"),
                    .target = xcfw_target,
                    .optimize = xcfw_optimize,
                }),
            });
            lib.bundle_compiler_rt = true;
            lib.root_module.addOptions("config", xcfw_options);
            lib.root_module.addImport("objc", objc_dep.module("objc"));
            lib.root_module.addImport("tracy", xcfw_tracy.module("tracy"));
            lib.root_module.addImport("dve", b.addModule("dve_xcfw", .{
                .root_source_file = b.path("src/root.zig"),
                .imports = &.{
                    .{ .name = "config", .module = xcfw_options.createModule() },
                    .{ .name = "objc", .module = objc_dep.module("objc") },
                    .{ .name = "tracy", .module = xcfw_tracy.module("tracy") },
                },
            }));
            lib.root_module.linkFramework("NaturalLanguage", .{});
            lib.root_module.linkFramework("CoreML", .{});
            lib.root_module.linkFramework("Foundation", .{});
            lib.root_module.link_libc = true;
            // Set the install name at link time so install_name_tool is not needed.
            lib.install_name = "@rpath/DVECore.framework/DVECore";
            libs[i] = lib.getEmittedBin();
        }

        // lipo: merge arm64 + x86_64 into a universal dylib
        const lipo = RunStep.create(b, "lipo DVECore");
        lipo.addArgs(&.{ "lipo", "-create", "-output" });
        const universal = lipo.addOutputFileArg("DVECore");
        lipo.addFileArg(libs[0]);
        lipo.addFileArg(libs[1]);

        // Assemble DVECore.framework: dylib + headers + model resources
        const fw_out = "zig-out/DVECore.framework";
        const xcfw_out = "zig-out/DVECore.xcframework";

        const rm = RunStep.create(b, "rm DVECore artifacts");
        rm.addArgs(&.{ "rm", "-rf", fw_out, xcfw_out });

        const mk_fw = RunStep.create(b, "construct DVECore.framework");
        mk_fw.has_side_effects = true;
        mk_fw.addArgs(&.{ "/bin/sh", "scripts/mk-framework.sh" });
        mk_fw.addFileArg(universal);
        mk_fw.addFileArg(mpnet_model_path);
        mk_fw.addFileArg(mpnet_tokenizer_path);
        mk_fw.addArg(fw_out);
        mk_fw.step.dependOn(&lipo.step);
        mk_fw.step.dependOn(&rm.step);

        // xcodebuild -create-xcframework from the assembled framework
        const xcfw = RunStep.create(b, "xcodebuild xcframework");
        xcfw.has_side_effects = true;
        xcfw.addArgs(&.{ "xcodebuild", "-create-xcframework" });
        xcfw.addArg("-framework");
        xcfw.addArg(fw_out);
        xcfw.addArg("-output");
        xcfw.addArg(xcfw_out);
        xcfw.step.dependOn(&mk_fw.step);

        xcfw_step.dependOn(&xcfw.step);
    }
}

/// Installs the mpnet runtime assets (model + tokenizer) into `b`'s install prefix
/// under `share/`, where `MpnetEmbedder` looks for them by default
/// (`<exe_dir>/../share/`). Call this from your own build.zig:
///
///     const dve_dep = b.dependency("dve", .{ .target = target, .optimize = optimize });
///     @import("dve").installModels(b, dve_dep);
///
/// Note `@import("dve")` resolves to this build script, not to the `dve` module --
/// inside build.zig a dependency name refers to its build.zig struct.
///
/// Only needed if you use the `mpnet_embedding` model; `apple_nlembedding` is
/// served by the OS and requires no model files.
pub fn installModels(b: *std.Build, dve_dep: *std.Build.Dependency) void {
    const coreml_models = dve_dep.builder.dependency("coreml_models", .{});
    for (addModelInstalls(b, coreml_models)) |s| b.getInstallStep().dependOn(s);
}

/// Creates install steps for the mpnet model + tokenizer, owned by `b`. Returns the
/// steps so callers can attach them to whichever step needs the assets present (the
/// install step, or a specific test's run step).
fn addModelInstalls(b: *std.Build, coreml_models: *std.Build.Dependency) [2]*Step {
    const install_model = b.addInstallDirectory(.{
        .source_dir = coreml_models.path("all_mpnet_base_v2/all_mpnet_base_v2.mlpackage"),
        .install_dir = .{ .custom = "share" },
        .install_subdir = "all_mpnet_base_v2.mlpackage",
    });
    const install_tokenizer = b.addInstallFile(
        coreml_models.path("all_mpnet_base_v2/tokenizer.json"),
        "share/tokenizer.json",
    );
    return .{ &install_model.step, &install_tokenizer.step };
}

/// src/llama_bridge.c as a static library of its own, linked against a prebuilt
/// llama.cpp. Building it as one artifact means the public `dve` module and the
/// in-repo targets share a single compile of the bridge, and the libllama link,
/// its search path and its rpath travel to whatever links it.
///
/// src/llama.zig compiles to a stub unless -Dllama is set, so nothing here runs
/// for the default build and llama.cpp does not need to be present.
fn llamaBridgeLib(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    llama_root: []const u8,
    llama_model_path: []const u8,
) LlamaBridge {
    const lib_dir = b.pathJoin(&.{ llama_root, "build", "bin" });
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "dve_llama_bridge",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    lib.root_module.addCSourceFile(.{
        .file = b.path("src/llama_bridge.c"),
        .flags = &.{
            "-std=c11",
            "-Wall",
            "-Wextra",
            b.fmt("-DDVE_LLAMA_MODEL_PATH=\"{s}\"", .{llama_model_path}),
        },
    });
    lib.root_module.addIncludePath(b.path("src"));
    lib.root_module.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ llama_root, "include" }),
    });
    lib.root_module.addIncludePath(.{
        .cwd_relative = b.pathJoin(&.{ llama_root, "ggml", "include" }),
    });
    lib.root_module.addLibraryPath(.{ .cwd_relative = lib_dir });
    lib.root_module.linkSystemLibrary("llama", .{});
    return .{ .lib = lib, .lib_dir = lib_dir };
}

/// A built bridge, and the one call that attaches it to a module.
const LlamaBridge = struct {
    lib: *Step.Compile,
    lib_dir: []const u8,

    fn link(self: LlamaBridge, m: *std.Build.Module) void {
        m.linkLibrary(self.lib);
        // `-lllama` rides along with the archive, but a static library does not
        // propagate its search path: the linker has to be told again where to
        // find libllama, and dyld where to find it and the libggml-*.dylib it
        // pulls in -- cmake leaves both wherever it built them.
        m.addLibraryPath(.{ .cwd_relative = self.lib_dir });
        m.addRPath(.{ .cwd_relative = self.lib_dir });
    }
};

const StorageQuantize = enum { none, f_16, i_8 };

const std = @import("std");
const Step = std.Build.Step;
const RunStep = Step.Run;
