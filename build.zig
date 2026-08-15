pub fn build(b: *std.Build) !void {
    const debug = b.option(bool, "debug-output", "Show debug output") orelse false;
    const embedding_model = b.option(
        EmbeddingModel,
        "embedding-model",
        "Embedding model to use",
    ) orelse .mpnet_embedding;
    const test_filter: ?[]const u8 = b.option(
        []const u8,
        "test-filter",
        "Filter to select specific tests",
    );
    const use_lldb = b.option(bool, "lldb", "Run tests under lldb debugger") orelse false;
    // Root of a built llama.cpp checkout: needs include/, ggml/include/ and the
    // shared libraries in build/bin/. Only consulted when the llama embedding
    // model is selected.
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

    const real_vec_sz: usize = switch (embedding_model) {
        .apple_nlembedding => 512,
        .mpnet_embedding => 768,
        .llama_nomic_embed_text_v1_5_f32 => 768,
    };

    ///////////////////////
    // Mpnet Model Fetch //
    ///////////////////////
    const coreml_models = b.dependency("coreml_models", .{});
    const mpnet_model_path = coreml_models.path("all_mpnet_base_v2/all_mpnet_base_v2.mlpackage");
    const mpnet_tokenizer_path = coreml_models.path("all_mpnet_base_v2/tokenizer.json");

    // When mpnet is selected, install the runtime assets (model + tokenizer)
    // to zig-out/share/ so the exe can find them at their default relative paths.
    // Consumers wire in these assets with one line:
    //   b.getInstallStep().dependOn(dve_dep.builder.getInstallStep());
    if (embedding_model == .mpnet_embedding) {
        const install_model = b.addInstallDirectory(.{
            .source_dir = mpnet_model_path,
            .install_dir = .{ .custom = "share" },
            .install_subdir = "all_mpnet_base_v2.mlpackage",
        });
        const install_tokenizer = b.addInstallFile(
            mpnet_tokenizer_path,
            "share/tokenizer.json",
        );
        b.getInstallStep().dependOn(&install_model.step);
        b.getInstallStep().dependOn(&install_tokenizer.step);
    }

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
    real_options.addOption(usize, "vec_sz", real_vec_sz);
    real_options.addOption(bool, "debug", debug);
    real_options.addOption(EmbeddingModel, "embedding_model", embedding_model);

    // Fake config used for storage/util tests that don't need real embeddings.
    const fake_options = b.addOptions();
    fake_options.addOption(usize, "vec_sz", @as(usize, 3));
    fake_options.addOption(bool, "debug", debug);
    fake_options.addOption(EmbeddingModel, "embedding_model", EmbeddingModel.apple_nlembedding);

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

    // Wires up ObjC + tracy imports, framework/lib links, and -- when the llama
    // embedding model is selected -- the llama.cpp C bridge.
    const deps = Deps{
        .b = b,
        .cfg = real_options,
        .objc = objc_dep,
        .tracy = tracy_dep,
        .tracy_enable = tracy_enable,
        .llama = embedding_model == .llama_nomic_embed_text_v1_5_f32,
        .llama_root = llama_root,
        .llama_model_path = llama_model_path,
    };

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
        deps.add(t);
        const install_models = b.addInstallDirectory(.{
            .source_dir = mpnet_model_path,
            .install_dir = .{ .custom = "share" },
            .install_subdir = "all_mpnet_base_v2.mlpackage",
        });
        const install_tokenizer = b.addInstallFile(
            mpnet_tokenizer_path,
            "share/tokenizer.json",
        );
        const run = runTest(b, t, use_lldb);
        run.step.dependOn(&install_models.step);
        run.step.dependOn(&install_tokenizer.step);
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
        deps.add(t);
        const install_models = b.addInstallDirectory(.{
            .source_dir = mpnet_model_path,
            .install_dir = .{ .custom = "share" },
            .install_subdir = "all_mpnet_base_v2.mlpackage",
        });
        const install_tokenizer = b.addInstallFile(
            mpnet_tokenizer_path,
            "share/tokenizer.json",
        );
        const run = runTest(b, t, use_lldb);
        run.step.dependOn(&install_models.step);
        run.step.dependOn(&install_tokenizer.step);
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
        deps.add(t);
        const install_models = b.addInstallDirectory(.{
            .source_dir = mpnet_model_path,
            .install_dir = .{ .custom = "share" },
            .install_subdir = "all_mpnet_base_v2.mlpackage",
        });
        const install_tokenizer = b.addInstallFile(
            mpnet_tokenizer_path,
            "share/tokenizer.json",
        );
        const run = runTest(b, t, use_lldb);
        run.step.dependOn(&install_models.step);
        run.step.dependOn(&install_tokenizer.step);
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
        deps.add(t);
        const install_models = b.addInstallDirectory(.{
            .source_dir = mpnet_model_path,
            .install_dir = .{ .custom = "share" },
            .install_subdir = "all_mpnet_base_v2.mlpackage",
        });
        const install_tokenizer = b.addInstallFile(
            mpnet_tokenizer_path,
            "share/tokenizer.json",
        );
        const run = runTest(b, t, use_lldb);
        run.step.dependOn(&install_models.step);
        run.step.dependOn(&install_tokenizer.step);
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
        deps.add(exe);

        // Installed by this step only, so a plain `zig build` doesn't build the
        // harness. The embedder resolves model assets relative to the
        // executable, so the shared install step has to run too.
        wikitest_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        wikitest_step.dependOn(b.getInstallStep());
    }

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(test_vec_storage);
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
    // Always compiled with mpnet_embedding; model files are bundled in Resources/.
    const xcfw_step = b.step("xcframework", "Build DVECore.xcframework");
    {
        const arm_target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos });
        const x86_target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .macos });
        const xcfw_optimize: std.builtin.OptimizeMode = .ReleaseFast;

        const mpnet_options = b.addOptions();
        mpnet_options.addOption(usize, "vec_sz", @as(usize, 768));
        mpnet_options.addOption(bool, "debug", false);
        mpnet_options.addOption(EmbeddingModel, "embedding_model", EmbeddingModel.mpnet_embedding);

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
            lib.root_module.addOptions("config", mpnet_options);
            lib.root_module.addImport("objc", objc_dep.module("objc"));
            lib.root_module.addImport("tracy", xcfw_tracy.module("tracy"));
            lib.root_module.addImport("dve", b.addModule("dve_xcfw", .{
                .root_source_file = b.path("src/root.zig"),
                .imports = &.{
                    .{ .name = "config", .module = mpnet_options.createModule() },
                    .{ .name = "objc", .module = objc_dep.module("objc") },
                    .{ .name = "tracy", .module = xcfw_tracy.module("tracy") },
                },
            }));
            lib.root_module.linkFramework("NaturalLanguage", .{});
            lib.root_module.linkFramework("CoreML", .{});
            lib.root_module.linkFramework("Foundation", .{});
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

const EmbeddingModel = enum {
    apple_nlembedding,
    mpnet_embedding,
    llama_nomic_embed_text_v1_5_f32,
};

/// The imports and links every artifact built against the real config needs.
const Deps = struct {
    b: *std.Build,
    cfg: *Step.Options,
    objc: *std.Build.Dependency,
    tracy: *std.Build.Dependency,
    tracy_enable: bool,
    llama: bool,
    llama_root: []const u8,
    llama_model_path: []const u8,

    fn add(self: Deps, t: *Step.Compile) void {
        t.root_module.addOptions("config", self.cfg);
        t.root_module.addImport("objc", self.objc.module("objc"));
        t.root_module.addImport("tracy", self.tracy.module("tracy"));
        t.root_module.linkFramework("NaturalLanguage", .{});
        t.root_module.linkFramework("CoreML", .{});
        t.root_module.linkFramework("Foundation", .{});
        if (self.tracy_enable) {
            t.root_module.linkLibrary(self.tracy.artifact("tracy"));
            t.root_module.link_libcpp = true;
        }
        if (self.llama) self.addLlama(t);
    }

    /// Compiles src/llama_bridge.c into `t` and links it against a prebuilt
    /// llama.cpp. src/llama.zig only references the bridge when the llama model
    /// is selected, so this is skipped entirely for the other models and
    /// llama.cpp does not need to be present to build them.
    fn addLlama(self: Deps, t: *Step.Compile) void {
        const b = self.b;
        const lib_dir = b.pathJoin(&.{ self.llama_root, "build", "bin" });
        t.root_module.addCSourceFile(.{
            .file = b.path("src/llama_bridge.c"),
            .flags = &.{
                "-std=c11",
                "-Wall",
                "-Wextra",
                b.fmt("-DDVE_LLAMA_MODEL_PATH=\"{s}\"", .{self.llama_model_path}),
            },
        });
        t.root_module.addIncludePath(b.path("src"));
        t.root_module.addIncludePath(.{
            .cwd_relative = b.pathJoin(&.{ self.llama_root, "include" }),
        });
        t.root_module.addIncludePath(.{
            .cwd_relative = b.pathJoin(&.{ self.llama_root, "ggml", "include" }),
        });
        t.root_module.addLibraryPath(.{ .cwd_relative = lib_dir });
        // libllama.dylib and the libggml-*.dylib it pulls in stay wherever cmake
        // left them, so bake that lookup path into the binary.
        t.root_module.addRPath(.{ .cwd_relative = lib_dir });
        t.root_module.linkSystemLibrary("llama", .{});
        t.root_module.link_libc = true;
    }
};

const std = @import("std");
const Step = std.Build.Step;
const RunStep = Step.Run;
