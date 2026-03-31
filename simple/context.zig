const std = @import("std");
const zml = @import("zml");

/// Manages the shared execution state, memory allocation, and RNG for the benchmarking suite.
pub const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    rng: std.Random.DefaultPrng,

    /// Initializes a new benchmarking Context, configuring the ZML platform and RNG.
    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Context {
        return .{
            .allocator = allocator,
            .io = io,
            .platform = try .auto(allocator, io, .{}),
            .rng = std.Random.DefaultPrng.init(0),
        };
    }

    /// Frees resources associated with the Context, including the ZML platform.
    pub fn deinit(self: *Context) void {
        self.platform.deinit(self.allocator);
    }

    /// Returns a PRNG source for initializing benchmark tensors.
    pub fn random(self: *Context) std.Random {
        return self.rng.random();
    }

    /// Allocates and initializes a slice of elements using a provided factory function.
    pub fn allocAndInit(self: *Context, comptime T: type, len: usize, factory: fn (std.Random) T) ![]T {
        const data = try self.allocator.alloc(T, len);
        const r = self.random();
        for (data) |*v| v.* = factory(r);
        return data;
    }

    /// Creates a ZML Buffer object from a Zig slice of memory.
    pub fn bufferFromSlice(self: Context, shape: zml.Shape, data: anytype) !zml.Buffer {
        const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);
        return zml.Buffer.fromSlice(self.io, self.platform, zml.Slice.init(shape, std.mem.sliceAsBytes(data)), replicated_sharding);
    }

    /// Copies contents from a ZML Buffer back into a Zig slice.
    pub fn sliceFromBuffer(self: Context, buffer: zml.Buffer, out: anytype) !void {
        try buffer.toSlice(self.io, zml.Slice.init(buffer.shape(), std.mem.sliceAsBytes(out)));
    }
};
