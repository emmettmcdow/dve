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

    // Install mpnet model files into this project's zig-out/share/,
    // where the exe will find them at their default relative paths.
    const coreml_models = dve_dep.builder.dependency("coreml_models", .{});
    const install_model = b.addInstallDirectory(.{
        .source_dir = coreml_models.path("all_mpnet_base_v2/all_mpnet_base_v2.mlpackage"),
        .install_dir = .{ .custom = "share" },
        .install_subdir = "all_mpnet_base_v2.mlpackage",
    });
    const install_tokenizer = b.addInstallFile(
        coreml_models.path("all_mpnet_base_v2/tokenizer.json"),
        "share/tokenizer.json",
    );
    b.getInstallStep().dependOn(&install_model.step);
    b.getInstallStep().dependOn(&install_tokenizer.step);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    const run_step = b.step("run", "Run dve-repl");
    run_step.dependOn(&run.step);
}
