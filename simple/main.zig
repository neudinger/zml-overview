const std = @import("std");
const Context = @import("context.zig").Context;
const bench = @import("benchmarks.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    var ctx = try Context.init(init.gpa, init.io);
    defer ctx.deinit();

    try bench.saxpy(&ctx);
    try bench.matmul(&ctx);
    try bench.mod_matmul(&ctx);
    try bench.heat_transfer(&ctx);
    try bench.black_scholes(&ctx);
    // https://pressio.github.io/pressio-demoapps/euler_2d_kelvin_helmholtz.html
    // TODO: implement this if i get many likes
    // try bench.euler_2d_kelvin_helmholtz(&ctx);
}
