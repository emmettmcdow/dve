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
        .name = "embed-watch",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("dve", dve_module);
    exe.root_module.linkFramework("NaturalLanguage", .{});
    exe.root_module.linkFramework("CoreML", .{});
    exe.root_module.linkFramework("Foundation", .{});

    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    const run_step = b.step("run", "Run embed-watch");
    run_step.dependOn(&run.step);
}
