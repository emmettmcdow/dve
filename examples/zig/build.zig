const std = @import("std");

/// Mirrors dve's EmbeddingModel tag names so the choice can be forwarded to
/// src/main.zig by name.
const EmbeddingModel = enum {
    apple_nlembedding,
    mpnet_embedding,
    llama_nomic_embed_text_v1_5_f32,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const model = b.option(
        EmbeddingModel,
        "model",
        "Embedding model to demo (default: mpnet_embedding)",
    ) orelse .mpnet_embedding;
    const llama = model == .llama_nomic_embed_text_v1_5_f32;
    // Only consulted for the llama model. dve derives the .gguf path from this,
    // and DVE_LLAMA_MODEL overrides that at run time.
    const llama_path = b.option(
        []const u8,
        "llama-path",
        "Path to a built llama.cpp checkout (default: $HOME/llama.cpp)",
    ) orelse b.pathJoin(&.{ std.posix.getenv("HOME") orelse ".", "llama.cpp" });

    const dve_dep = b.dependency("dve", .{
        .target = target,
        .optimize = optimize,
        // Which model to use is a call-site choice, but the llama.cpp backend
        // has to be linked at build time for that choice to be available.
        .llama = llama,
        .@"llama-path" = llama_path,
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

    // Hands the model choice to src/main.zig, which selects the VectorEngine.
    const opts = b.addOptions();
    opts.addOption(EmbeddingModel, "model", model);
    exe.root_module.addOptions("example_config", opts);

    b.installArtifact(exe);

    // Install mpnet model files into this project's zig-out/share/,
    // where the exe will find them at their default relative paths.
    // The other two models need no model files installed: apple_nlembedding is
    // served by the OS, and the llama gguf is loaded from its own path.
    if (model == .mpnet_embedding) @import("dve").installModels(b, dve_dep);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    const run_step = b.step("run", "Run dve-repl");
    run_step.dependOn(&run.step);
}
