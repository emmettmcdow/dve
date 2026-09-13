pub const ValidationError = error{NotL2};

/// Filename fragment identifying which storage type a database on disk holds. Quantized
/// vectors share neither the element type nor the byte width of unquantized ones, so each
/// setting gets its own files: rebuilding under a different -Dstorage-quantize leaves the
/// other's database intact instead of colliding with it.
///
/// `-v2` marks the `vstore.zig` format. It shares nothing with the `vec_storage.zig` file
/// that used to live under the unsuffixed name, and opening one as the other is a hard
/// `IncompatibleDatabase`. Giving the new format its own name turns that failure into a
/// rebuild -- an upgraded install opens an empty store and re-indexes -- and leaves the old
/// file untouched for a migration that may never be written.
pub const db_suffix = switch (config.quant) {
    .none => "-v2",
    .f_16 => "-f16-v2",
    .i_8 => "-i8-v2",
};

/// Components of an L2-normalized vector have an RMS of exactly 1/sqrt(N), so the range
/// worth encoding is fixed by N alone and no per-vector scale has to be stored alongside
/// the data. CLAMP_SIGMA picks how much of the tail survives: measured across real mpnet
/// and NLEmbedding output the largest component seen was 5.2 sigma, so 6 saturates nothing
/// while keeping the step as small as possible.
const CLAMP_SIGMA: f32 = 6.0;

/// Multiplier taking an f32 component into the i8 domain. Its inverse square undoes the
/// scaling of a dot product between two quantized vectors.
pub fn i8Scale(comptime N: usize) f32 {
    return 127.0 * @sqrt(@as(f32, @floatFromInt(N))) / CLAMP_SIGMA;
}

/// Narrows an f32 vector to f16 for storage.
pub fn quant32to16(comptime N: usize, v: @Vector(N, f32)) @Vector(N, f16) {
    return @floatCast(v);
}

/// Narrows an f32 vector to i8 for storage, saturating past CLAMP_SIGMA rather than
/// wrapping. Lossy and not exactly invertible; `storedDot` undoes the scaling.
pub fn quant32toi8(comptime N: usize, v: @Vector(N, f32)) @Vector(N, i8) {
    const scale: @Vector(N, f32) = @splat(i8Scale(N));
    const lo: @Vector(N, f32) = @splat(-127.0);
    const hi: @Vector(N, f32) = @splat(127.0);
    return @intFromFloat(@round(@min(@max(v * scale, lo), hi)));
}

/// Dot product of two stored vectors, in the units of the f32 embeddings they came from,
/// whatever type they are stored as. Always accumulates wider than the storage type: an
/// f16 running total swallows half of each addend well before N terms, and an i8 product
/// overflows i8 on the very first one.
pub fn storedDot(comptime N: usize, comptime T: type, a: @Vector(N, T), b: @Vector(N, T)) f32 {
    switch (@typeInfo(T)) {
        .float => {
            const wa: @Vector(N, f32) = @floatCast(a);
            const wb: @Vector(N, f32) = @floatCast(b);
            return @reduce(.Add, wa * wb);
        },
        .int => {
            // Chunked into 64 lanes rather than one N-wide multiply because zig 0.15.2
            // mis-lowers overflow-checked integer vector arithmetic at wide
            // non-power-of-two widths: @Vector(768, i32) traps "integer overflow" on
            // values as small as 3. Float vectors carry no such check, which is why only
            // the quantized path ever hit it. Worst case here is 768*128*128 = 1.26e7,
            // about 170x under i32.
            const LANES = 64;
            const av: [N]T = a;
            const bv: [N]T = b;
            var raw: i32 = 0;
            var i: usize = 0;
            while (i + LANES <= N) : (i += LANES) {
                const ca: @Vector(LANES, i32) = @intCast(@as(@Vector(LANES, T), av[i..][0..LANES].*));
                const cb: @Vector(LANES, i32) = @intCast(@as(@Vector(LANES, T), bv[i..][0..LANES].*));
                raw += @reduce(.Add, ca * cb);
            }
            while (i < N) : (i += 1) raw += @as(i32, av[i]) * @as(i32, bv[i]);

            const scale = i8Scale(N);
            return @as(f32, @floatFromInt(raw)) / (scale * scale);
        },
        else => @compileError("no dot product for stored type " ++ @typeName(T)),
    }
}

/// `storedDot` over operands that stay where they are. Zig pads `@Vector(768, f32)` to 4096
/// bytes, so passing two of them by value copies 8 KB per call -- twice the bytes a whole-store
/// scan reads out of the file. That makes the copies, not the data, the thing that bounds the
/// scan, which is the wrong thing to be bounded by. This reads through the pointers in
/// fixed-width lanes instead.
///
/// Summation order differs from `storedDot`'s full-width `@reduce`, so the two can disagree in
/// the last ulp or so. Both are approximations of the same sum and neither is the "true" one;
/// nothing here compares similarities for exact equality.
pub fn storedDotAt(comptime N: usize, comptime T: type, a: *const [N]T, b: *const [N]T) f32 {
    // 16 lanes is four NEON registers' worth of f32, wide enough to keep the unit busy and
    // narrow enough that the accumulator stays in registers across the loop.
    const LANES = 16;
    switch (@typeInfo(T)) {
        .float => {
            var acc: @Vector(LANES, f32) = @splat(0);
            var i: usize = 0;
            while (i + LANES <= N) : (i += LANES) {
                const va: @Vector(LANES, f32) = @floatCast(@as(@Vector(LANES, T), a[i..][0..LANES].*));
                const vb: @Vector(LANES, f32) = @floatCast(@as(@Vector(LANES, T), b[i..][0..LANES].*));
                acc += va * vb;
            }
            var total = @reduce(.Add, acc);
            while (i < N) : (i += 1) {
                total += @as(f32, @floatCast(a[i])) * @as(f32, @floatCast(b[i]));
            }
            return total;
        },
        .int => {
            // Same widening and the same reason for it as `storedDot`'s integer path: an i8
            // product overflows i8 on the first term, and zig 0.15.2 mis-lowers wide
            // non-power-of-two integer vector arithmetic.
            var acc: @Vector(LANES, i32) = @splat(0);
            var i: usize = 0;
            while (i + LANES <= N) : (i += LANES) {
                const va: @Vector(LANES, i32) = @intCast(@as(@Vector(LANES, T), a[i..][0..LANES].*));
                const vb: @Vector(LANES, i32) = @intCast(@as(@Vector(LANES, T), b[i..][0..LANES].*));
                acc += va * vb;
            }
            var raw: i32 = @reduce(.Add, acc);
            while (i < N) : (i += 1) raw += @as(i32, a[i]) * @as(i32, b[i]);

            const scale = i8Scale(N);
            return @as(f32, @floatFromInt(raw)) / (scale * scale);
        },
        else => @compileError("no dot product for stored type " ++ @typeName(T)),
    }
}

/// How far a stored unit vector's norm may drift from 1.0 purely by being stored as T.
fn l2Tolerance(comptime N: usize, comptime T: type) f32 {
    switch (@typeInfo(T)) {
        // Rounding each component to T perturbs it by at most half an ulp, which bounds the
        // relative error of the norm by the same fraction -- so a unit vector stored as f16
        // cannot come back closer than ~4.9e-4. f32 keeps the historical 1e-5, which is far
        // looser than its own 6e-8 bound and so stays exactly as strict as it has been.
        .float => return @max(1e-5, std.math.floatEps(T) / 2),
        // Quantization error is uniform across one step, leaving the norm near 1.0 with a
        // systematic +N/(24*scale^2) lift from the error's own magnitude and a random part
        // of standard deviation 1/(scale*sqrt(12)). Five sigma still catches real
        // corruption while passing every correctly quantized vector.
        .int => {
            const scale = i8Scale(N);
            const sigma = 1.0 / (scale * @sqrt(12.0));
            const bias = @as(f32, @floatFromInt(N)) / (24.0 * scale * scale);
            return 5.0 * sigma + bias;
        },
        else => @compileError("no L2 tolerance for stored type " ++ @typeName(T)),
    }
}

pub fn validateL2(comptime N: usize, comptime T: type, v: @Vector(N, T)) ValidationError!void {
    // storedDot both widens the accumulator and undoes any quantization scaling, so this is
    // the norm of the vector the caller originally handed to the engine.
    const norm = @sqrt(storedDot(N, T, v, v));
    if (@abs(norm - 1.0) >= l2Tolerance(N, T)) {
        std.debug.print("norm: {d:.8} (expected ~1.0)\n", .{norm});
        return ValidationError.NotL2;
    }
}

const config = @import("config");
const std = @import("std");
