const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dve_dep = b.dependency("dve", .{
        .target = target,
        .optimize = optimize,
        .@"embedding-model" = .mpnet_embedding,
    });
    const dve_module = dve_dep.module("dve");

    const exe = b.addExecutable(.{
        .name = "dve-repl",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("dve", dve_module);
    exe.root_module.linkFramework("NaturalLanguage", .{});
    exe.root_module.linkFramework("CoreML", .{});
    exe.root_module.linkFramework("Foundation", .{});

    b.installArtifact(exe);

    // Install mpnet model files from dve's source tree into this project's zig-out/share/,
    // where the exe will find them at their default relative paths.
    // Depend on dve's install step to ensure model generation completes first.
    const dve_install = dve_dep.builder.getInstallStep();
    const install_model = b.addInstallDirectory(.{
        .source_dir = dve_dep.path("models/all_mpnet_base_v2/all_mpnet_base_v2.mlpackage"),
        .install_dir = .{ .custom = "share" },
        .install_subdir = "all_mpnet_base_v2.mlpackage",
    });
    install_model.step.dependOn(dve_install);
    const install_tokenizer = b.addInstallFile(
        dve_dep.path("models/all_mpnet_base_v2/tokenizer.json"),
        "share/tokenizer.json",
    );
    install_tokenizer.step.dependOn(dve_install);
    b.getInstallStep().dependOn(&install_model.step);
    b.getInstallStep().dependOn(&install_tokenizer.step);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    const run_step = b.step("run", "Run dve-repl");
    run_step.dependOn(&run.step);
}
