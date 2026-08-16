//! Scaling harness for dve against the Simple Wikipedia corpus.
//!
//! Modes:
//!   embed   ingest articles from a corpus directory into a dve database
//!   search  run queries against an existing database
//!   stat    report counts and on-disk size for an existing database
//!
//! The full corpus is far larger than dve currently handles, so `--limit` caps
//! the number of articles ingested. Articles are visited in sorted filename
//! order, so `--limit 1000` is always a prefix of `--limit 10000` -- ingesting
//! successively larger tiers compares like with like.
//!
//! The interesting column in `embed` output is docs/s(int) -- the rate over the
//! last progress interval. Cumulative rate hides super-linear degradation;
//! interval rate shows it falling off a cliff.

const std = @import("std");
const builtin = @import("builtin");
const dve = @import("dve");

/// dve logs one line per embedded document at info level. At corpus scale that
/// output dwarfs the harness's own, so raise the threshold.
pub const std_options: std.Options = .{ .log_level = .warn };

const VectorEngine = dve.VectorEngine(dve.embedding_model);
const Embedder = switch (dve.embedding_model) {
    .apple_nlembedding => dve.embed.NLEmbedder,
    .mpnet_embedding => dve.embed.MpnetEmbedder,
};

const MAX_ARTICLE_BYTES: usize = 64 * 1024 * 1024;

const default_queries = [_][]const u8{
    "who was the first president of the united states",
    "how do volcanoes erupt",
    "the largest planet in the solar system",
    "world war two started in which year",
    "photosynthesis in plants",
    "capital city of france",
};

const Mode = enum { embed, search, stat };

const Options = struct {
    mode: Mode,
    corpus: []const u8 = "wikidata/md",
    db: []const u8 = "wikitest-db",
    /// 0 means no limit.
    limit: usize = 0,
    progress: usize = 100,
    k: usize = 10,
    repeat: usize = 3,
    unique: bool = false,
    csv: ?[]const u8 = null,
    queries: []const []const u8 = &default_queries,
};

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const opts = try parseArgs(allocator, args);
    defer if (opts.queries.ptr != &default_queries) allocator.free(opts.queries);

    switch (opts.mode) {
        .embed => try runEmbed(allocator, opts),
        .search => try runSearch(allocator, opts),
        .stat => try runStat(allocator, opts),
    }
}

// ******************************************************************************************* Modes

fn runEmbed(allocator: std.mem.Allocator, opts: Options) !void {
    var corpus = std.fs.cwd().openDir(opts.corpus, .{ .iterate = true }) catch |err| {
        fatal("cannot open corpus dir '{s}': {t}\n" ++
            "(run ./download.sh first, or pass --corpus <dir>)", .{ opts.corpus, err });
    };
    defer corpus.close();

    const names = try collectArticles(allocator, corpus, opts.limit);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }
    if (names.len == 0) fatal("no files found in '{s}'", .{opts.corpus});

    var db_dir = try std.fs.cwd().makeOpenPath(opts.db, .{ .iterate = true });
    defer db_dir.close();

    var csv = try Csv.open(opts.csv);
    defer csv.close();

    std.debug.print(
        "model      {s} ({d} dims)\ncorpus     {s} ({d} articles selected)\ndatabase   {s}\n\n",
        .{ @tagName(dve.embedding_model), dve.types.vec_sz, opts.corpus, names.len, opts.db },
    );

    var load_timer = try std.time.Timer.start();
    const engine = try initEngine(allocator, db_dir);
    defer engine.deinit();
    const db = engine.db;
    const load_ns = load_timer.read();

    if (db.vec_storage.vec_n != 0) {
        std.debug.print(
            "note: database already holds {d} vectors; this run appends to it\n\n",
            .{db.vec_storage.vec_n},
        );
    }
    std.debug.print("opened existing database in {D}\n\n", .{load_ns});

    printHeader();

    var total_timer = try std.time.Timer.start();
    var interval_timer = try std.time.Timer.start();
    var interval_start_vecs = db.vec_storage.vec_n;
    var done: usize = 0;
    var skipped: usize = 0;

    outer: for (names) |name| {
        const contents = corpus.readFileAlloc(allocator, name, MAX_ARTICLE_BYTES) catch |err| {
            std.debug.print("skip {s}: {t}\n", .{ name, err });
            skipped += 1;
            continue;
        };
        defer allocator.free(contents);

        // Reading files is far faster than embedding them, so a full queue is the steady
        // state here rather than an error -- wait for a slot instead of dropping the
        // article. Retrying without the sleep spins the producer at full speed through
        // dupe/copy/free and contends on the queue locks, starving the embedder thread.
        inner: while (true) {
            db.embedTextAsync(name, contents) catch |err| switch (err) {
                error.Full => {
                    std.Thread.sleep(250 * std.time.ns_per_us);
                    continue :inner;
                },
                else => {
                    std.debug.print("skip {s}: {t}\n", .{ name, err });
                    skipped += 1;
                    continue :outer;
                },
            };
            break :inner;
        }
        done += 1;

        if (done % opts.progress == 0 or done == names.len) {
            const sample = Sample{
                .articles = done,
                .vectors = db.vec_storage.vec_n,
                .interval_articles = opts.progress,
                .interval_vectors = db.vec_storage.vec_n - interval_start_vecs,
                .interval_ns = interval_timer.lap(),
                .total_ns = total_timer.read(),
                .rss_bytes = rssBytes(),
                .db_bytes = dirSize(db_dir) catch 0,
            };
            printSample(sample);
            try csv.write(sample);
            interval_start_vecs = db.vec_storage.vec_n;
        }
    }
    std.debug.print("Waiting for background embedder to complete...\n", .{});
    db.shutdown();

    const elapsed = total_timer.read();
    std.debug.print(
        "\nembedded {d} articles ({d} skipped) -> {d} vectors in {D}\n" ++
            "capacity {d} slots, {d} paths, {f} on disk, {f} peak rss\n",
        .{
            done,
            skipped,
            db.vec_storage.vec_n,
            elapsed,
            db.vec_storage.meta.capacity,
            db.note_id_map.count(),
            fmtBytes(dirSize(db_dir) catch 0),
            fmtBytes(rssBytes()),
        },
    );
}

fn runSearch(allocator: std.mem.Allocator, opts: Options) !void {
    var db_dir = std.fs.cwd().openDir(opts.db, .{ .iterate = true }) catch |err| {
        fatal("cannot open database '{s}': {t}\n(run the embed mode first)", .{ opts.db, err });
    };
    defer db_dir.close();

    var load_timer = try std.time.Timer.start();
    const engine = try initEngine(allocator, db_dir);
    defer engine.deinit();
    const db = engine.db;
    const load_ns = load_timer.read();

    std.debug.print(
        "database   {s} ({d} vectors, {d} paths, {f} on disk)\nload       {D}\n" ++
            "search     {s}, k={d}, {d} repeats per query\n\n",
        .{
            opts.db,
            db.vec_storage.vec_n,
            db.note_id_map.count(),
            fmtBytes(dirSize(db_dir) catch 0),
            load_ns,
            if (opts.unique) "uniqueSearch" else "search",
            opts.k,
            opts.repeat,
        },
    );

    const buf = try allocator.alloc(dve.SearchResult, opts.k);
    defer allocator.free(buf);
    const samples = try allocator.alloc(u64, opts.repeat);
    defer allocator.free(samples);

    for (opts.queries) |query| {
        var found: usize = 0;
        for (samples) |*sample| {
            var timer = try std.time.Timer.start();
            found = if (opts.unique)
                try db.uniqueSearch(query, buf)
            else
                try db.search(query, buf);
            sample.* = timer.read();
        }
        std.mem.sort(u64, samples, {}, std.sort.asc(u64));

        std.debug.print("\"{s}\"\n  {d} results | min {D} | median {D} | max {D}\n", .{
            query,
            found,
            samples[0],
            samples[samples.len / 2],
            samples[samples.len - 1],
        });
        for (buf[0..found]) |r| {
            std.debug.print("    {d:.4}  {s} [{d}..{d}]\n", .{
                r.similarity,
                r.path,
                r.start_i,
                r.end_i,
            });
        }
        std.debug.print("\n", .{});
    }
}

fn runStat(allocator: std.mem.Allocator, opts: Options) !void {
    var db_dir = std.fs.cwd().openDir(opts.db, .{ .iterate = true }) catch |err| {
        fatal("cannot open database '{s}': {t}", .{ opts.db, err });
    };
    defer db_dir.close();

    var load_timer = try std.time.Timer.start();
    const engine = try initEngine(allocator, db_dir);
    defer engine.deinit();
    const db = engine.db;
    const load_ns = load_timer.read();

    std.debug.print(
        "model      {s} ({d} dims)\nvectors    {d} occupied / {d} capacity\npaths      {d}\n" ++
            "load       {D}\npeak rss   {f}\n\nfiles:\n",
        .{
            @tagName(dve.embedding_model),
            dve.types.vec_sz,
            db.vec_storage.vec_n,
            db.vec_storage.meta.capacity,
            db.note_id_map.count(),
            load_ns,
            fmtBytes(rssBytes()),
        },
    );

    var total: u64 = 0;
    var it = db_dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        const st = try db_dir.statFile(entry.name);
        total += st.size;
        std.debug.print("  {s:<28} {f}\n", .{ entry.name, fmtBytes(st.size) });
    }
    std.debug.print("  {s:<28} {f}\n", .{ "total", fmtBytes(total) });
}

// ******************************************************************************************* Setup

/// The engine takes ownership of the embedder's *contents* (`deinit` calls
/// through the interface) but not of its allocation, so the caller has to keep
/// the pointer around and destroy it after `db.deinit()`.
const Engine = struct {
    db: *VectorEngine,
    embedder: *Embedder,
    allocator: std.mem.Allocator,

    fn deinit(self: Engine) void {
        self.db.deinit();
        self.allocator.destroy(self.embedder);
    }
};

fn initEngine(allocator: std.mem.Allocator, db_dir: std.fs.Dir) !Engine {
    const e = try allocator.create(Embedder);
    errdefer allocator.destroy(e);
    e.* = switch (dve.embedding_model) {
        .mpnet_embedding => try dve.embed.MpnetEmbedder.init(.{}),
        .apple_nlembedding => try dve.embed.NLEmbedder.init(),
    };
    return .{
        .db = try VectorEngine.init(allocator, db_dir, e.embedder()),
        .embedder = e,
        .allocator = allocator,
    };
}

/// Collects up to `limit` filenames from `dir`, sorted, so that a smaller limit
/// always yields a prefix of a larger one.
fn collectArticles(allocator: std.mem.Allocator, dir: std.fs.Dir, limit: usize) ![][]u8 {
    var names: std.ArrayList([]u8) = .{};
    errdefer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        try names.append(allocator, try allocator.dupe(u8, entry.name));
    }

    const items = try names.toOwnedSlice(allocator);
    std.mem.sort([]u8, items, {}, struct {
        fn lessThan(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    if (limit == 0 or limit >= items.len) return items;

    for (items[limit..]) |n| allocator.free(n);
    return try allocator.realloc(items, limit);
}

// ***************************************************************************************** Metrics

const Sample = struct {
    articles: usize,
    vectors: usize,
    interval_articles: usize,
    interval_vectors: usize,
    interval_ns: u64,
    total_ns: u64,
    rss_bytes: u64,
    db_bytes: u64,
};

fn perSecond(count: usize, ns: u64) f64 {
    if (ns == 0) return 0;
    return @as(f64, @floatFromInt(count)) / (@as(f64, @floatFromInt(ns)) / std.time.ns_per_s);
}

fn printHeader() void {
    std.debug.print("{s:>10} {s:>10} {s:>10} {s:>12} {s:>12} {s:>10} {s:>10}\n", .{
        "articles", "vectors", "elapsed", "docs/s(int)", "docs/s(avg)", "rss", "db size",
    });
}

fn printSample(s: Sample) void {
    std.debug.print("{d:>10} {d:>10} {D:>10} {d:>12.1} {d:>12.1} {f:>10} {f:>10}\n", .{
        s.articles,
        s.vectors,
        s.total_ns,
        perSecond(s.interval_articles, s.interval_ns),
        perSecond(s.articles, s.total_ns),
        fmtBytes(s.rss_bytes),
        fmtBytes(s.db_bytes),
    });
}

/// Optional CSV sink so interval samples can be plotted.
const Csv = struct {
    file: ?std.fs.File = null,
    writer: std.fs.File.Writer = undefined,
    buf: [4096]u8 = undefined,

    fn open(path: ?[]const u8) !Csv {
        const p = path orelse return .{};
        var self = Csv{ .file = try std.fs.cwd().createFile(p, .{}) };
        self.writer = self.file.?.writer(&self.buf);
        try self.writer.interface.writeAll(
            "articles,vectors,total_ns,interval_ns,interval_articles," ++
                "interval_vectors,docs_per_s_interval,docs_per_s_avg,rss_bytes,db_bytes\n",
        );
        return self;
    }

    fn write(self: *Csv, s: Sample) !void {
        if (self.file == null) return;
        try self.writer.interface.print("{d},{d},{d},{d},{d},{d},{d:.3},{d:.3},{d},{d}\n", .{
            s.articles,
            s.vectors,
            s.total_ns,
            s.interval_ns,
            s.interval_articles,
            s.interval_vectors,
            perSecond(s.interval_articles, s.interval_ns),
            perSecond(s.articles, s.total_ns),
            s.rss_bytes,
            s.db_bytes,
        });
    }

    fn close(self: *Csv) void {
        if (self.file) |f| {
            self.writer.interface.flush() catch {};
            f.close();
        }
    }
};

fn rssBytes() u64 {
    const ru = std.posix.getrusage(std.posix.rusage.SELF);
    const maxrss: u64 = @intCast(@max(ru.maxrss, 0));
    // Darwin reports maxrss in bytes; everyone else in kilobytes.
    return switch (builtin.os.tag) {
        .macos, .ios, .tvos, .watchos, .visionos => maxrss,
        else => maxrss * 1024,
    };
}

fn dirSize(dir: std.fs.Dir) !u64 {
    var total: u64 = 0;
    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        const st = dir.statFile(entry.name) catch continue;
        total += st.size;
    }
    return total;
}

const ByteSize = struct {
    bytes: u64,

    pub fn format(self: ByteSize, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
        var value: f64 = @floatFromInt(self.bytes);
        var unit: usize = 0;
        while (value >= 1024 and unit < units.len - 1) : (unit += 1) value /= 1024;
        if (unit == 0) return w.print("{d}B", .{self.bytes});
        return w.print("{d:.1}{s}", .{ value, units[unit] });
    }
};

fn fmtBytes(bytes: u64) ByteSize {
    return .{ .bytes = bytes };
}

// ********************************************************************************** Argument parsing

fn parseArgs(allocator: std.mem.Allocator, args: [][:0]u8) !Options {
    if (args.len < 2) usage(1);

    const mode = std.meta.stringToEnum(Mode, args[1]) orelse {
        if (std.mem.eql(u8, args[1], "-h") or std.mem.eql(u8, args[1], "--help")) usage(0);
        fatal("unknown mode '{s}' (expected embed, search, or stat)", .{args[1]});
    };

    var opts = Options{ .mode = mode };
    var queries: std.ArrayList([]const u8) = .{};
    errdefer queries.deinit(allocator);

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--corpus")) {
            opts.corpus = nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--db")) {
            opts.db = nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--limit")) {
            opts.limit = try parseUsize(nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "--progress")) {
            opts.progress = try parseUsize(nextArg(args, &i));
            if (opts.progress == 0) fatal("--progress must be greater than 0", .{});
        } else if (std.mem.eql(u8, arg, "-k")) {
            opts.k = try parseUsize(nextArg(args, &i));
            if (opts.k == 0) fatal("-k must be greater than 0", .{});
        } else if (std.mem.eql(u8, arg, "--repeat")) {
            opts.repeat = try parseUsize(nextArg(args, &i));
            if (opts.repeat == 0) fatal("--repeat must be greater than 0", .{});
        } else if (std.mem.eql(u8, arg, "--unique")) {
            opts.unique = true;
        } else if (std.mem.eql(u8, arg, "--csv")) {
            opts.csv = nextArg(args, &i);
        } else if (std.mem.eql(u8, arg, "--query")) {
            try queries.append(allocator, nextArg(args, &i));
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            usage(0);
        } else {
            fatal("unknown option '{s}' (try --help)", .{arg});
        }
    }

    if (queries.items.len > 0) {
        opts.queries = try queries.toOwnedSlice(allocator);
    } else {
        queries.deinit(allocator);
    }
    return opts;
}

fn nextArg(args: [][:0]u8, i: *usize) []const u8 {
    i.* += 1;
    if (i.* >= args.len) fatal("'{s}' requires a value", .{args[i.* - 1]});
    return args[i.*];
}

fn parseUsize(s: []const u8) !usize {
    return std.fmt.parseInt(usize, s, 10) catch fatal("'{s}' is not a number", .{s});
}

fn fatal(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

fn usage(code: u8) noreturn {
    std.debug.print(
        \\usage: wikitest <mode> [options]
        \\
        \\modes:
        \\  embed     ingest articles from a corpus directory into a dve database
        \\  search    run queries against an existing database
        \\  stat      report counts and on-disk size for an existing database
        \\
        \\options:
        \\  --corpus <dir>   article directory        (default: wikidata/md)
        \\  --db <dir>       database directory       (default: wikitest-db)
        \\  --limit <n>      max articles to ingest, 0 for all   (default: 0)
        \\  --progress <n>   report every n articles  (default: 100)
        \\  --csv <path>     also write interval samples as CSV
        \\  --query <text>   query to run, repeatable (default: a built-in set)
        \\  -k <n>           results per query        (default: 10)
        \\  --repeat <n>     timing repeats per query (default: 3)
        \\  --unique         use uniqueSearch instead of search
        \\
        \\examples:
        \\  wikitest embed --limit 1000 --csv tier-1k.csv
        \\  wikitest search --query "how do volcanoes erupt" -k 5
        \\  wikitest stat
        \\
    , .{});
    std.process.exit(code);
}
