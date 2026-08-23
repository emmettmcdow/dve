pub fn UniqueCircularBuffer(T: type, ID_T: type, GET_ID_FN: fn (T) ID_T) type {
    const HashMap = HashMapUnmanaged(ID_T, usize, AutoContext(ID_T), 99);

    return struct {
        N: u32,
        ring_buf: []T,
        id_to_idx: *HashMap,
        allocator: Allocator,
        read_i: usize = 0,
        write_i: usize = 0,
        mutex: Mutex = Mutex{},

        pub const Error = error{Full};

        pub fn init(allocator: Allocator, sz: u32) !*@This() {
            var map = try allocator.create(HashMap);
            map.* = .empty;
            try map.ensureTotalCapacity(allocator, sz);
            const self = try allocator.create(@This());
            self.* = .{
                .N = sz,
                .ring_buf = try allocator.alloc(T, sz),
                .id_to_idx = map,
                .allocator = allocator,
            };
            return self;
        }

        pub fn deinit(self: *@This()) void {
            self.id_to_idx.deinit(self.allocator);
            self.allocator.destroy(self.id_to_idx);
            self.allocator.free(self.ring_buf);
            self.allocator.destroy(self);
        }

        /// Pop from the front of the queue.
        pub fn pop(self: *@This()) ?T {
            const output = b: {
                self.mutex.lock();
                defer self.mutex.unlock();

                if (self.read_i == self.write_i) {
                    return null;
                }
                defer self.read_i = (self.read_i + 1) % self.N;
                const item = self.ring_buf[self.read_i];
                assert(self.id_to_idx.remove(GET_ID_FN(item)));
                break :b item;
            };
            return output;
        }

        /// Push to the back of the queue.
        ///
        /// An item whose id is already queued replaces the queued one in place, keeping its
        /// position. The replaced item is returned so the caller can release anything it
        /// owns; dropping the return value leaks it.
        pub fn push(self: *@This(), item: T) Error!?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            // Check if we can update rather than add.
            if (self.id_to_idx.getEntry(GET_ID_FN(item))) |entry| {
                const displaced = self.ring_buf[entry.value_ptr.*];
                self.ring_buf[entry.value_ptr.*] = item;
                return displaced;
            }

            if ((self.write_i + 1) % self.N == self.read_i) {
                return error.Full;
            }
            self.ring_buf[self.write_i] = item;
            self.id_to_idx.putAssumeCapacity(GET_ID_FN(item), self.write_i);
            self.write_i = (self.write_i + 1) % self.N;
            return null;
        }
    };
}

fn usizeID(a: usize) usize {
    return a;
}

test "UniqueCircularBuffer" {
    const capacity = 4;
    const UsizeCircularBuf = UniqueCircularBuffer(usize, usize, usizeID);
    const allocator = std.testing.allocator;

    { // FIFO base
        var buf = try UsizeCircularBuf.init(allocator, capacity);
        defer buf.deinit();
        for (0..capacity - 1) |i| try expectEqual(null, try buf.push(i));
        for (0..capacity - 1) |i| try expectEqual(i, buf.pop());
    }
    { // Error Cases
        var buf = try UsizeCircularBuf.init(allocator, capacity);
        defer buf.deinit();
        try expectEqual(null, buf.pop());
        for (0..capacity - 1) |i| _ = try buf.push(i);
        try expectEqual(UsizeCircularBuf.Error.Full, buf.push(4));
    }
    { // Update unique.
        const TestStruct = struct {
            id: usize,
            val: usize,

            pub fn getID(self: @This()) usize {
                return self.id;
            }
        };
        const StructCircularBuf = UniqueCircularBuffer(TestStruct, usize, TestStruct.getID);
        var buf = try StructCircularBuf.init(allocator, capacity);
        defer buf.deinit();

        const a = TestStruct{ .id = 1, .val = 1 };
        try expectEqual(null, try buf.push(a));
        const b = TestStruct{ .id = 2, .val = 2 };
        try expectEqual(null, try buf.push(b));
        const c = TestStruct{ .id = 3, .val = 3 };
        try expectEqual(null, try buf.push(c));

        const want = 420;
        const b_mod = TestStruct{ .id = b.id, .val = want };
        // The displaced item comes back so its owner can free what it holds.
        try expectEqualDeep(b, try buf.push(b_mod));

        try expectEqualDeep(a, buf.pop());
        try expectEqualDeep(b_mod, buf.pop());
        try expectEqualDeep(c, buf.pop());
    }
    { // A replacement does not consume a slot.
        var buf = try UsizeCircularBuf.init(allocator, capacity);
        defer buf.deinit();
        for (0..capacity - 1) |i| _ = try buf.push(i);
        try expectEqual(0, try buf.push(0));
        try expectEqual(UsizeCircularBuf.Error.Full, buf.push(capacity));
    }
}

pub const TrackingAllocator = struct {
    const K = usize;
    const V = usize;
    const HashMap = HashMapUnmanaged(K, V, AutoContext(K), 50);
    const StrHashMap = HashMapUnmanaged(K, []const u8, AutoContext(K), 50);
    const Self = @This();

    parent: std.mem.Allocator,
    /// Total allocated from each callsite
    caller_map: *HashMap,
    stack_info: *StrHashMap,
    bytes_allocated: usize = 0,
    peak_bytes: usize = 0,

    pub fn init(base_allocator: std.mem.Allocator) !*Self {
        const map = try base_allocator.create(HashMap);
        map.* = .empty;
        const stack_info = try base_allocator.create(StrHashMap);
        stack_info.* = .empty;
        const self = try base_allocator.create(Self);
        self.* = .{
            .parent = base_allocator,
            .caller_map = map,
            .stack_info = stack_info,
        };
        return self;
    }

    pub fn deinit(self: *TrackingAllocator) void {
        self.caller_map.deinit(self.parent);
        self.parent.destroy(self.caller_map);
        var stack_it = self.stack_info.iterator();
        while (stack_it.next()) |e| {
            self.parent.free(e.value_ptr.*);
        }
        self.stack_info.deinit(self.parent);
        self.parent.destroy(self.stack_info);
        self.parent.destroy(self);
        return;
    }

    pub fn allocator(self: *TrackingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = TAalloc,
                .resize = TAresize,
                .free = TAfree,
                .remap = TAremap,
            },
        };
    }

    fn TAalloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
        var self: *Self = @ptrCast(@alignCast(ctx));
        const result = self.parent.rawAlloc(len, alignment, ret_addr);
        if (result != null) {
            self.bytes_allocated += len;
            if (self.bytes_allocated > self.peak_bytes) {
                self.peak_bytes = self.bytes_allocated;
            }
            if (self.caller_map.get(ret_addr)) |sz| {
                self.caller_map.put(self.parent, ret_addr, sz + len) catch unreachable;
            } else {
                self.caller_map.put(self.parent, ret_addr, len) catch unreachable;
                self.TAdumpStack(ret_addr);
            }
        }
        return result;
    }

    fn TAresize(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
        var self: *Self = @ptrCast(@alignCast(ctx));
        const result = self.parent.rawResize(memory, alignment, new_len, ret_addr);
        if (result) self.TAresizeAdj(new_len, memory.len, ret_addr);
        return result;
    }

    fn TAfree(ctx: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
        var self: *Self = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(memory, alignment, ret_addr);
        self.bytes_allocated -= memory.len;
        return;
    }

    fn TAremap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        var self: *Self = @ptrCast(@alignCast(ctx));
        const result = self.parent.rawRemap(memory, alignment, new_len, ret_addr);
        if (result != null) self.TAresizeAdj(new_len, memory.len, ret_addr);
        return result;
    }

    fn TAresizeAdj(self: *Self, new_len: usize, original_len: usize, ret_addr: usize) void {
        if (new_len > original_len) {
            const bytes_added = new_len - original_len;
            if (self.caller_map.get(ret_addr)) |sz| {
                self.caller_map.put(self.parent, ret_addr, sz + bytes_added) catch unreachable;
            } else {
                self.caller_map.put(self.parent, ret_addr, bytes_added) catch unreachable;
                self.TAdumpStack(ret_addr);
            }
            self.bytes_allocated += bytes_added;
            if (self.bytes_allocated > self.peak_bytes) {
                self.peak_bytes = self.bytes_allocated;
            }
        } else if (new_len < original_len) {
            self.bytes_allocated -= original_len - new_len;
        }
    }

    pub fn TAreport(self: *Self) void {
        const total_kb_val = self.peak_bytes >> 10;
        const total_mb_val = total_kb_val >> 10;
        if (total_mb_val != 0) {
            std.debug.print("Peak memory usage: {d}MB\n", .{total_mb_val});
        } else if (total_kb_val != 0) {
            std.debug.print("Peak memory usage: {d}KB\n", .{total_kb_val});
        } else {
            std.debug.print("Peak memory usage: {d}B\n", .{self.peak_bytes});
        }

        const n_items = self.caller_map.size;
        const entry = struct {
            sz: usize,
            addr: usize,

            const InnerSelf = @This();
            pub fn order(_: void, a: InnerSelf, b: InnerSelf) bool {
                return std.math.order(a.sz, b.sz) == std.math.Order.gt;
            }
        };
        var items = self.parent.alloc(entry, n_items) catch unreachable;
        defer self.parent.free(items);

        var it = self.caller_map.iterator();
        var i: usize = 0;
        while (it.next()) |item| {
            items[i] = .{ .sz = item.value_ptr.*, .addr = item.key_ptr.* };
            i += 1;
        }

        std.sort.insertion(entry, items, {}, entry.order);
        for (items) |item| {
            const kb_val = item.sz >> 10;
            const mb_val = kb_val >> 10;
            if (mb_val != 0) {
                std.debug.print("    {d}MB used from 0x{x}\n", .{ mb_val, item.addr });
                std.debug.print("{s}", .{self.stack_info.get(item.addr).?});
            } else if (kb_val != 0) {
                std.debug.print("    {d}KB used from 0x{x}\n", .{ kb_val, item.addr });
                std.debug.print("{s}", .{self.stack_info.get(item.addr).?});
            } else {
                std.debug.print("    {d}B used from 0x{x}\n", .{ item.sz, item.addr });
            }
        }
    }

    fn TAdumpStack(self: *Self, ret_addr: usize) void {
        var aw = std.io.Writer.Allocating.init(self.parent);
        defer aw.deinit();

        const debug_info = std.debug.getSelfDebugInfo() catch unreachable;
        var stack_it = std.debug.StackIterator.init(null, null);
        var started_printing = false;
        o: while (stack_it.next()) |address| {
            const module = debug_info.getModuleForAddress(address) catch break;
            const symbol_info = module.getSymbolAtAddress(debug_info.allocator, address) catch break;
            const sl = symbol_info.source_location orelse continue;
            if (!started_printing) {
                if (std.mem.indexOf(u8, sl.file_name, "dve/src") != null) { // Our source code
                    const our_fns = [_][]const u8{ "TAalloc", "TAresize", "TAfree", "TAremap", "TAresizeAdj", "TAreport", "TAdumpStack" };
                    for (our_fns) |our_fn| if (std.mem.eql(u8, symbol_info.name, our_fn)) continue :o;
                    // dve/src and not
                } else { // no dve/src
                    continue;
                }
                started_printing = true;
            }
            aw.writer.print("        {s} @ {s}:{d}\n", .{ symbol_info.name, sl.file_name, sl.line }) catch unreachable;
        }
        const stack = aw.toOwnedSlice() catch unreachable;
        self.stack_info.put(self.parent, ret_addr, stack) catch unreachable;
    }
};

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;
const AutoContext = std.hash_map.AutoContext;
const expectEqual = std.testing.expectEqual;
const expectEqualDeep = std.testing.expectEqualDeep;
const HashMapUnmanaged = std.hash_map.HashMapUnmanaged;
const Mutex = std.Thread.Mutex;
