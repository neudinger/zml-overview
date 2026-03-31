const std = @import("std");
const zml = @import("zml");
const SimpleModel = @import("model.zig").SimpleModel;

pub fn main(init: std.process.Init) !void {
    var platform = try zml.Platform.auto(init.gpa, init.io, .{});
    defer platform.deinit(init.gpa);

    const model = SimpleModel{};

    const s: f32 = 100.0;
    const k: f32 = 100.0;
    const t: f32 = 1.0;
    const r: f32 = 0.05;
    const sigma: f32 = 0.2;

    const replicated_sharding: zml.sharding.Sharding = try zml.sharding.replicatedSharding(platform);
    var exe: zml.exe.Exe = try platform.compile(init.gpa, init.io, model, .black_scholes, .{
        zml.Tensor.init(.{}, .f32),
        zml.Tensor.init(.{}, .f32),
        zml.Tensor.init(.{}, .f32),
        zml.Tensor.init(.{}, .f32),
        zml.Tensor.init(.{}, .f32),
    }, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var s_buf: zml.Buffer = try zml.Buffer.fromSlice(init.io, platform, zml.Slice.init(zml.Shape.scalar(.f32), std.mem.asBytes(&s)), replicated_sharding);
    defer s_buf.deinit();
    var k_buf: zml.Buffer = try zml.Buffer.fromSlice(init.io, platform, zml.Slice.init(zml.Shape.scalar(.f32), std.mem.asBytes(&k)), replicated_sharding);
    defer k_buf.deinit();
    var t_buf: zml.Buffer = try zml.Buffer.fromSlice(init.io, platform, zml.Slice.init(zml.Shape.scalar(.f32), std.mem.asBytes(&t)), replicated_sharding);
    defer t_buf.deinit();
    var r_buf = try zml.Buffer.fromSlice(init.io, platform, zml.Slice.init(zml.Shape.scalar(.f32), std.mem.asBytes(&r)), replicated_sharding);
    defer r_buf.deinit();
    var sigma_buf = try zml.Buffer.fromSlice(init.io, platform, zml.Slice.init(zml.Shape.scalar(.f32), std.mem.asBytes(&sigma)), replicated_sharding);
    defer sigma_buf.deinit();

    var args: zml.exe.Exe.Arguments = try exe.args(init.gpa);
    defer args.deinit(init.gpa);
    args.set(.{ s_buf, k_buf, t_buf, r_buf, sigma_buf });

    var results: zml.exe.Exe.Results = try exe.results(init.gpa);
    defer results.deinit(init.gpa);
    exe.call(args, &results);

    const ResStruct = struct { call: zml.Buffer, put: zml.Buffer };
    var res_bufs: ResStruct = results.get(ResStruct);
    defer {
        res_bufs.call.deinit();
        res_bufs.put.deinit();
    }

    var call_out: [1]f32 = undefined;
    try res_bufs.call.toSlice(init.io, zml.Slice.init(res_bufs.call.shape(), std.mem.sliceAsBytes(call_out[0..])));
    var put_out: [1]f32 = undefined;
    try res_bufs.put.toSlice(init.io, zml.Slice.init(res_bufs.put.shape(), std.mem.sliceAsBytes(put_out[0..])));

    const call_val: f32 = call_out[0];
    const put_val: f32 = put_out[0];

    if (@abs(call_val - 10.4506) > 1e-3 or @abs(put_val - 5.5735) > 1e-3) {
        std.debug.print("zml black_scholes mismatch: call={d} put={d}\n", .{ call_val, put_val });
        return error.ValueMismatch;
    }

    std.debug.print("simple_test_zml passed: call={d} put={d}\n", .{ call_val, put_val });
}
