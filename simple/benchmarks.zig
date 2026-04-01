const std = @import("std");
const log = std.log;
const zml = @import("zml");

const Context = @import("context.zig").Context;
const SimpleModel = @import("model.zig").SimpleModel;
const ref = @import("reference.zig");
const opt = @import("optimized.zig");

/// Helper function to assert that two floating-point slices are approximately equal within an epsilon.
fn expectApproxEq(expected: []const f32, actual: []const f32, epsilon: f32) !void {
    if (expected.len != actual.len) return error.LengthMismatch;
    for (expected, actual) |e, a| {
        if (@abs(e - a) > epsilon) {
            log.err("Mismatch: expected {d}, got {d}", .{ e, a });
            return error.ValueMismatch;
        }
    }
}

/// Helper function to assert that two integer slices are exactly equal.
fn expectEq(expected: []const i32, actual: []const i32) !void {
    if (expected.len != actual.len) return error.LengthMismatch;
    for (expected, actual) |e, a| {
        if (e != a) {
            log.err("Mismatch: expected {d}, got {d}", .{ e, a });
            return error.ValueMismatch;
        }
    }
}

/// Benchmarks standard matrix multiplication comparing reference, optimized, and ZML implementations.
/// Formula: $$C_{i,j} = \sum_{k} A_{i,k} B_{k,j}$$
pub fn matmul(ctx: *Context) !void {
    const M: usize = 1024;
    const K: usize = 1024;
    const N: usize = 1024;
    log.info("Benchmarking MatMul ({d}x{d} x {d}x{d})...", .{ M, K, K, N });

    // Allocations
    const lhs_data: []const f32 = try ctx.allocAndInit(f32, M * K, struct {
        fn f(r: std.Random) f32 {
            return r.float(f32);
        }
    }.f);
    defer ctx.allocator.free(lhs_data);
    const rhs_data: []const f32 = try ctx.allocAndInit(f32, K * N, struct {
        fn f(r: std.Random) f32 {
            return r.float(f32);
        }
    }.f);
    defer ctx.allocator.free(rhs_data);
    const ref_out: []f32 = try ctx.allocator.alloc(f32, M * N);
    defer ctx.allocator.free(ref_out);
    const opt_out: []f32 = try ctx.allocator.alloc(f32, M * N);
    defer ctx.allocator.free(opt_out);

    // 1. Zig Reference
    var start: std.Io.Timestamp = std.Io.Clock.awake.now(ctx.io);
    ref.matmul(M, K, N, lhs_data, rhs_data, ref_out);
    const ref_ns: i96 = start.untilNow(ctx.io, .awake).toNanoseconds();

    // 2. Zig Optimized
    start = std.Io.Clock.awake.now(ctx.io);
    opt.matmul(M, K, N, lhs_data, rhs_data, opt_out);
    const opt_ns: i96 = start.untilNow(ctx.io, .awake).toNanoseconds();

    try expectApproxEq(ref_out, opt_out, 1e-3);

    // 3. ZML Execution
    const lhs_shape: zml.Shape = zml.Shape.init(.{ .m = M, .c = K }, .f32);
    const rhs_shape: zml.Shape = zml.Shape.init(.{ .c = K, .n = N }, .f32);

    const replicated_sharding: zml.sharding.Sharding = try zml.sharding.replicatedSharding(ctx.platform);
    var exe: zml.exe.Exe = try ctx.platform.compile(ctx.allocator, ctx.io, SimpleModel{}, .matmul, .{ zml.Tensor.fromShape(lhs_shape), zml.Tensor.fromShape(rhs_shape) }, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();
    // log.debug("Compiled ZML MatMul model", .{exe});

    var lhs_buf: zml.Buffer = try ctx.bufferFromSlice(lhs_shape, lhs_data);
    defer lhs_buf.deinit();
    var rhs_buf: zml.Buffer = try ctx.bufferFromSlice(rhs_shape, rhs_data);
    defer rhs_buf.deinit();

    var args: zml.exe.Exe.Arguments = try exe.args(ctx.allocator);
    defer args.deinit(ctx.allocator);
    args.set(.{ lhs_buf, rhs_buf });

    var results: zml.exe.Exe.Results = try exe.results(ctx.allocator);
    defer results.deinit(ctx.allocator);

    start = std.Io.Clock.awake.now(ctx.io);
    exe.call(args, &results);
    const zml_ns: i96 = start.untilNow(ctx.io, .awake).toNanoseconds();

    var res_buf: zml.Buffer = results.get(zml.Buffer);
    defer res_buf.deinit();

    const zml_out = try ctx.allocator.alloc(f32, M * N);
    defer ctx.allocator.free(zml_out);
    try ctx.sliceFromBuffer(res_buf, zml_out);

    try expectApproxEq(ref_out, zml_out, 1e-3);
    log.info("✅ MatMul Verified. Zig(Ref): {d:.2}ns, Zig(Opt): {d:.2}ns, ZML: {d:.2}ns", .{ ref_ns, opt_ns, zml_ns });
}

/// Benchmarks single-precision scalar multiplication and vector addition (SAXPY) comparing reference, optimized, and ZML implementations.
/// Formula: $$Z_i = a \cdot X_i + Y_i$$
pub fn saxpy(ctx: *Context) !void {
    const N: usize = 10_000_000;
    log.info("\nBenchmarking SAXPY ({d} elements)...", .{N});

    var a_val: f32 = 2.0;
    const x_data: []const f32 = try ctx.allocAndInit(f32, N, struct {
        fn f(r: std.Random) f32 {
            return r.float(f32);
        }
    }.f);
    defer ctx.allocator.free(x_data);
    const y_data: []const f32 = try ctx.allocAndInit(f32, N, struct {
        fn func(r: std.Random) f32 {
            return r.float(f32);
        }
    }.func);
    defer ctx.allocator.free(y_data);
    const ref_out: []f32 = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(ref_out);
    const opt_out: []f32 = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(opt_out);

    // 1. Zig Reference
    var start: std.Io.Timestamp = std.Io.Clock.awake.now(ctx.io);
    ref.saxpy(N, a_val, x_data, y_data, ref_out);
    const ref_ns: i96 = start.untilNow(ctx.io, .awake).toNanoseconds();

    // 2. Zig Optimized
    start = std.Io.Clock.awake.now(ctx.io);
    opt.saxpy(N, a_val, x_data, y_data, opt_out);
    const opt_ns: i96 = start.untilNow(ctx.io, .awake).toNanoseconds();

    try expectApproxEq(ref_out, opt_out, 1e-4);

    // 3. ZML Execution
    const shape: zml.Shape = zml.Shape.init(.{N}, .f32);
    const replicated_sharding: zml.sharding.Sharding = try zml.sharding.replicatedSharding(ctx.platform);
    var exe: zml.exe.Exe = try ctx.platform.compile(ctx.allocator, ctx.io, SimpleModel{}, .saxpy, .{
        zml.Tensor.init(.{}, .f32),
        zml.Tensor.fromShape(shape),
        zml.Tensor.fromShape(shape),
    }, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var a_buf: zml.Buffer = try ctx.bufferFromSlice(zml.Shape.scalar(.f32), std.mem.asBytes(&a_val));
    defer a_buf.deinit();
    var x_buf: zml.Buffer = try ctx.bufferFromSlice(shape, x_data);
    defer x_buf.deinit();
    var y_buf: zml.Buffer = try ctx.bufferFromSlice(shape, y_data);
    defer y_buf.deinit();

    var args: zml.exe.Exe.Arguments = try exe.args(ctx.allocator);
    defer args.deinit(ctx.allocator);
    args.set(.{ a_buf, x_buf, y_buf });

    var results: zml.exe.Exe.Results = try exe.results(ctx.allocator);
    defer results.deinit(ctx.allocator);

    const zml_out = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(zml_out);

    start = std.Io.Clock.awake.now(ctx.io);
    log.debug("Calling ZML SAXPY Exe with args: {any} at {}", .{ args, start });
    exe.call(args, &results);
    var res_buf: zml.Buffer = results.get(zml.Buffer);
    defer res_buf.deinit();
    const zml_ns: i96 = start.untilNow(ctx.io, .awake).toNanoseconds();

    try ctx.sliceFromBuffer(res_buf, zml_out);

    try expectApproxEq(ref_out, zml_out, 1e-4);
    log.info("✅ SAXPY Verified. Zig(Ref): {d:.2}ns, Zig(Opt): {d:.2}ns, ZML: {d:.2}ns", .{ ref_ns, opt_ns, zml_ns });
}

/// Benchmarks integer matrix multiplication with modulo $Q = 3329$ operation comparing reference, optimized, and ZML implementations.
/// Formula: $$C_{i,j} = \left( \sum_{k} A_{i,k} B_{k,j} \right) \pmod{Q}$$
pub fn mod_matmul(ctx: *Context) !void {
    const K: usize = 1024;
    log.info("\nBenchmarking ModMatMul ({d}x{d} x {d}x{d})...", .{ K, K, K, K });

    const lhs_data: []const i32 = try ctx.allocAndInit(i32, K * K, struct {
        fn f(r: std.Random) i32 {
            return @intCast(r.intRangeAtMost(i32, 0, 100));
        }
    }.f);
    defer ctx.allocator.free(lhs_data);
    const rhs_data: []const i32 = try ctx.allocAndInit(i32, K * K, struct {
        fn f(r: std.Random) i32 {
            return @intCast(r.intRangeAtMost(i32, 0, 100));
        }
    }.f);
    defer ctx.allocator.free(rhs_data);
    const ref_out: []i32 = try ctx.allocator.alloc(i32, K * K);
    defer ctx.allocator.free(ref_out);
    const opt_out: []i32 = try ctx.allocator.alloc(i32, K * K);
    defer ctx.allocator.free(opt_out);

    // 1. Zig Reference
    var start = std.Io.Clock.awake.now(ctx.io);
    ref.mod_matmul(K, K, K, lhs_data, rhs_data, ref_out);
    const ref_ns = start.untilNow(ctx.io, .awake).toNanoseconds();

    // 2. Zig Optimized
    start = std.Io.Clock.awake.now(ctx.io);
    opt.mod_matmul(K, K, K, lhs_data, rhs_data, opt_out);
    const opt_ns: i96 = start.untilNow(ctx.io, .awake).toNanoseconds();

    try expectEq(ref_out, opt_out);

    // 3. ZML Execution
    const lhs_shape: zml.Shape = zml.Shape.init(.{ .m = K, .c = K }, .i32);
    const rhs_shape: zml.Shape = zml.Shape.init(.{ .c = K, .n = K }, .i32);

    const replicated_sharding: zml.sharding.Sharding = try zml.sharding.replicatedSharding(ctx.platform);
    var exe: zml.exe.Exe = try ctx.platform.compile(ctx.allocator, ctx.io, SimpleModel{}, .mod_matmul, .{
        zml.Tensor.fromShape(lhs_shape),
        zml.Tensor.fromShape(rhs_shape),
    }, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var lhs_buf: zml.Buffer = try ctx.bufferFromSlice(lhs_shape, lhs_data);
    defer lhs_buf.deinit();
    var rhs_buf: zml.Buffer = try ctx.bufferFromSlice(rhs_shape, rhs_data);
    defer rhs_buf.deinit();

    var args: zml.exe.Exe.Arguments = try exe.args(ctx.allocator);
    defer args.deinit(ctx.allocator);
    args.set(.{ lhs_buf, rhs_buf });

    var results: zml.exe.Exe.Results = try exe.results(ctx.allocator);
    defer results.deinit(ctx.allocator);

    start = std.Io.Clock.awake.now(ctx.io);
    exe.call(args, &results);
    var res_buf: zml.Buffer = results.get(zml.Buffer);
    defer res_buf.deinit();
    const zml_ns: i96 = start.untilNow(ctx.io, .awake).toNanoseconds();

    const zml_out: []i32 = try ctx.allocator.alloc(i32, K * K);
    defer ctx.allocator.free(zml_out);
    try ctx.sliceFromBuffer(res_buf, zml_out);

    try expectEq(ref_out, zml_out);
    log.info("✅ ModMatMul Verified. Zig(Ref): {d:.2}ns, Zig(Opt): {d:.2}ns, ZML: {d:.2}ns", .{ ref_ns, opt_ns, zml_ns });
}

/// Benchmarks a 2D heat transfer simulation comparing reference, optimized, and ZML implementations.
/// Formula: $$U_{i,j}^{(t+1)} = \frac{1}{4} \left( U_{i-1,j}^{(t)} + U_{i+1,j}^{(t)} + U_{i,j-1}^{(t)} + U_{i,j+1}^{(t)} \right)$$
pub fn heat_transfer(ctx: *Context) !void {
    const H = 256;
    const W = 256;
    const steps = 100;
    log.info("\nBenchmarking Heat Transfer ({d}x{d}, {d} steps)...", .{ H, W, steps });

    // Init grid: Top row 100, others 0.
    const grid_init = try ctx.allocator.alloc(f32, H * W);
    defer ctx.allocator.free(grid_init);
    @memset(grid_init, 0);
    for (0..W) |j| grid_init[j] = 100.0;

    // Buffers for ping-pong
    const ref_grid = try ctx.allocator.dupe(f32, grid_init);
    defer ctx.allocator.free(ref_grid);
    const ref_out = try ctx.allocator.alloc(f32, H * W);
    defer ctx.allocator.free(ref_out);

    const opt_grid = try ctx.allocator.dupe(f32, grid_init);
    defer ctx.allocator.free(opt_grid);
    const opt_out = try ctx.allocator.alloc(f32, H * W);
    defer ctx.allocator.free(opt_out);

    // 1. Zig Reference
    var start = std.Io.Clock.awake.now(ctx.io);
    for (0..steps) |_| {
        ref.heat_transfer(H, W, ref_grid, ref_out);
        @memcpy(ref_grid, ref_out);
    }
    const ref_ns = start.untilNow(ctx.io, .awake).toNanoseconds();

    // 2. Zig Optimized
    start = std.Io.Clock.awake.now(ctx.io);
    for (0..steps) |_| {
        opt.heat_transfer(H, W, opt_grid, opt_out);
        @memcpy(opt_grid, opt_out);
    }
    const opt_ns = start.untilNow(ctx.io, .awake).toNanoseconds();

    try expectApproxEq(ref_grid, opt_grid, 1e-4);

    // 3. ZML
    const shape = zml.Shape.init(.{ .h = H, .w = W }, .f32);
    const replicated_sharding = try zml.sharding.replicatedSharding(ctx.platform);
    var exe = try ctx.platform.compile(ctx.allocator, ctx.io, SimpleModel{}, .heat_transfer_steps, .{
        zml.Tensor.fromShape(shape),
        zml.Tensor.init(.{}, .i32),
    }, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var args = try exe.args(ctx.allocator);
    defer args.deinit(ctx.allocator);
    var results = try exe.results(ctx.allocator);
    defer results.deinit(ctx.allocator);

    var zml_in_buf = try ctx.bufferFromSlice(shape, grid_init);
    defer zml_in_buf.deinit();
    var steps_i32: i32 = @intCast(steps);
    var steps_buf = try ctx.bufferFromSlice(zml.Shape.scalar(.i32), std.mem.asBytes(&steps_i32));
    defer steps_buf.deinit();

    start = std.Io.Clock.awake.now(ctx.io);
    args.set(.{ zml_in_buf, steps_buf });
    exe.call(args, &results);
    var final_buf: zml.Buffer = results.get(zml.Buffer);
    const zml_ns = start.untilNow(ctx.io, .awake).toNanoseconds();
    defer final_buf.deinit();

    const zml_out = try ctx.allocator.alloc(f32, H * W);
    defer ctx.allocator.free(zml_out);
    try ctx.sliceFromBuffer(final_buf, zml_out);

    try expectApproxEq(ref_grid, zml_out, 1e-3);
    log.info("✅ Heat Transfer Verified. Zig(Ref): {d:.2}ns, Zig(Opt): {d:.2}ns, ZML: {d:.2}ns", .{ ref_ns, opt_ns, zml_ns });
}

/// Benchmarks the Black-Scholes option pricing model comparing reference, optimized, and ZML implementations.
/// Formulas for European call:
/// $$C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$
/// $$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}$$
/// $$d_2 = d_1 - \sigma \sqrt{T}$$
pub fn black_scholes(ctx: *Context) !void {
    const N: usize = 1_000_000;
    log.info("\nBenchmarking Black-Scholes ({d} options)...", .{N});

    const s = try ctx.allocAndInit(f32, N, struct {
        fn f(r: std.Random) f32 {
            return 100.0 + r.float(f32) * 50.0;
        }
    }.f);
    defer ctx.allocator.free(s);
    const k = try ctx.allocAndInit(f32, N, struct {
        fn f(r: std.Random) f32 {
            return 100.0 + r.float(f32) * 50.0;
        }
    }.f);
    defer ctx.allocator.free(k);
    const t = try ctx.allocAndInit(f32, N, struct {
        fn f(r: std.Random) f32 {
            return 0.5 + r.float(f32) * 2.0;
        }
    }.f);
    defer ctx.allocator.free(t);
    const r_val = try ctx.allocAndInit(f32, N, struct {
        fn f(r: std.Random) f32 {
            return 0.01 + r.float(f32) * 0.05;
        }
    }.f);
    defer ctx.allocator.free(r_val);
    const sigma = try ctx.allocAndInit(f32, N, struct {
        fn f(r: std.Random) f32 {
            return 0.1 + r.float(f32) * 0.3;
        }
    }.f);
    defer ctx.allocator.free(sigma);

    const ref_call = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(ref_call);
    const ref_put = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(ref_put);

    const opt_call = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(opt_call);
    const opt_put = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(opt_put);

    // 1. Zig Reference
    var start = std.Io.Clock.awake.now(ctx.io);
    ref.black_scholes(N, s, k, t, r_val, sigma, ref_call, ref_put);
    const ref_ns = start.untilNow(ctx.io, .awake).toNanoseconds();

    // 2. Zig Optimized
    start = std.Io.Clock.awake.now(ctx.io);
    opt.black_scholes(N, s, k, t, r_val, sigma, opt_call, opt_put);
    const opt_ns = start.untilNow(ctx.io, .awake).toNanoseconds();

    try expectApproxEq(ref_call, opt_call, 1e-3);
    try expectApproxEq(ref_put, opt_put, 1e-3);

    // 3. ZML
    const shape = zml.Shape.init(.{N}, .f32);
    const replicated_sharding = try zml.sharding.replicatedSharding(ctx.platform);
    var exe = try ctx.platform.compile(ctx.allocator, ctx.io, SimpleModel{}, .black_scholes, .{
        zml.Tensor.fromShape(shape),
        zml.Tensor.fromShape(shape),
        zml.Tensor.fromShape(shape),
        zml.Tensor.fromShape(shape),
        zml.Tensor.fromShape(shape),
    }, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var s_buf = try ctx.bufferFromSlice(shape, s);
    defer s_buf.deinit();
    var k_buf = try ctx.bufferFromSlice(shape, k);
    defer k_buf.deinit();
    var t_buf = try ctx.bufferFromSlice(shape, t);
    defer t_buf.deinit();
    var r_buf = try ctx.bufferFromSlice(shape, r_val);
    defer r_buf.deinit();
    var sigma_buf = try ctx.bufferFromSlice(shape, sigma);
    defer sigma_buf.deinit();

    var args = try exe.args(ctx.allocator);
    defer args.deinit(ctx.allocator);
    args.set(.{ s_buf, k_buf, t_buf, r_buf, sigma_buf });

    var results = try exe.results(ctx.allocator);
    defer results.deinit(ctx.allocator);

    start = std.Io.Clock.awake.now(ctx.io);
    exe.call(args, &results);
    const zml_ns = start.untilNow(ctx.io, .awake).toNanoseconds();

    const ResStruct = struct { call: zml.Buffer, put: zml.Buffer };
    var res_bufs = results.get(ResStruct);
    defer {
        res_bufs.call.deinit();
        res_bufs.put.deinit();
    }

    const zml_call = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(zml_call);
    try ctx.sliceFromBuffer(res_bufs.call, zml_call);

    const zml_put = try ctx.allocator.alloc(f32, N);
    defer ctx.allocator.free(zml_put);
    try ctx.sliceFromBuffer(res_bufs.put, zml_put);

    try expectApproxEq(ref_call, zml_call, 1e-3);
    try expectApproxEq(ref_put, zml_put, 1e-3);

    log.info("✅ Black-Scholes Verified. Zig(Ref): {d:.2}ns, Zig(Opt): {d:.2}ns, ZML: {d:.2}ns", .{ ref_ns, opt_ns, zml_ns });
}
