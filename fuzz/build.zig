const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dve_dep = b.dependency("dve", .{
        .target = target,
        .optimize = optimize,
    });
    const dve_module = dve_dep.module("dve");

    const exe = b.addExecutable(.{
        .name = "dve-fuzz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("dve", dve_module);

    b.installArtifact(exe);

    // Install mpnet model files into this project's zig-out/share/,
    // where the exe will find them at their default relative paths.
    @import("dve").installModels(b, dve_dep, .{ .mpnet_embedding = true });

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    run.addPassthruArgs();
    const run_step = b.step("run", "Run dve-fuzz");
    run_step.dependOn(&run.step);
}
