const std = @import("std");

const max_threads = 16;

fn getNumThreads() usize {
    return std.Thread.getCpuCount() catch 1;
}

/// Multi-threaded single-precision scalar multiplication and vector addition.
/// Formula: $$Z_i = a \cdot X_i + Y_i$$
pub fn saxpy(n: usize, a: f32, x: []const f32, y: []const f32, out: []f32) void {
    const num_threads = @min(getNumThreads(), max_threads);
    if (num_threads <= 1 or n < 1024) {
        saxpyKernel(a, x[0..n], y[0..n], out[0..n]);
        return;
    }

    const chunk_size: usize = (n + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const start: usize = t * chunk_size;
        if (start >= n) break;
        const end: usize = @min(start + chunk_size, n);
        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(a_: f32, x_: []const f32, y_: []const f32, out_: []f32) void {
                saxpyKernel(a_, x_, y_, out_);
            }
        }.worker, .{ a, x[start..end], y[start..end], out[start..end] }) catch {
            // Fallback: do this chunk on the current thread
            saxpyKernel(a, x[start..end], y[start..end], out[start..end]);
            continue;
        };
        spawned += 1;
    }

    for (threads[0..spawned]) |t| t.join();
}

/// Vectorized kernel for single-precision scalar multiplication and vector addition.
/// Formula: $$Z_i = a \cdot X_i + Y_i$$
fn saxpyKernel(a: f32, x: []const f32, y: []const f32, out: []f32) void {
    const VL: usize = 64;
    const Vec: type = @Vector(VL, f32);
    const len: usize = x.len;

    var i: usize = 0;
    while (i + VL <= len) : (i += VL) {
        const x_vec: Vec = x[i..][0..VL].*;
        const y_vec: Vec = y[i..][0..VL].*;
        const a_vec: Vec = @splat(a);
        out[i..][0..VL].* = x_vec * a_vec + y_vec;
    }
    while (i < len) : (i += 1) {
        out[i] = a * x[i] + y[i];
    }
}

/// Multi-threaded standard matrix multiplication.
/// Formula: $$C_{i,j} = \sum_{k} A_{i,k} B_{k,j}$$
pub fn matmul(M: usize, K: usize, N: usize, lhs: []const f32, rhs: []const f32, out: []f32) void {
    const MC: usize = 64;
    const num_threads: usize = @min(getNumThreads(), max_threads);

    @memset(out, 0);

    if (num_threads <= 1 or M < MC) {
        matmulKernel(0, M, K, N, lhs, rhs, out);
        return;
    }

    // Each thread gets a range of row-blocks
    const num_row_blocks: usize = (M + MC - 1) / MC;
    const blocks_per_thread: usize = (num_row_blocks + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const block_start: usize = t * blocks_per_thread;
        if (block_start >= num_row_blocks) break;
        const block_end: usize = @min(block_start + blocks_per_thread, num_row_blocks);
        const m_start: usize = block_start * MC;
        const m_end: usize = @min(block_end * MC, M);

        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(m_s: usize, m_e: usize, K_: usize, N_: usize, lhs_: []const f32, rhs_: []const f32, out_: []f32) void {
                matmulKernel(m_s, m_e, K_, N_, lhs_, rhs_, out_);
            }
        }.worker, .{ m_start, m_end, K, N, lhs, rhs, out }) catch {
            matmulKernel(m_start, m_end, K, N, lhs, rhs, out);
            continue;
        };
        spawned += 1;
    }

    for (threads[0..spawned]) |t| t.join();
}

/// Blocked kernel for standard matrix multiplication.
/// Formula: $$C_{i,j} = \sum_{k} A_{i,k} B_{k,j}$$
fn matmulKernel(m_start: usize, m_end: usize, K: usize, N: usize, lhs: []const f32, rhs: []const f32, out: []f32) void {
    const NC: usize = 64;
    const KC: usize = 64;

    var n: usize = 0;
    while (n < N) : (n += NC) {
        const n_end: usize = @min(N, n + NC);
        var k: usize = 0;
        while (k < K) : (k += KC) {
            const k_end = @min(K, k + KC);
            for (m_start..m_end) |i| {
                for (n..n_end) |j| {
                    var sum: f32 = out[i * N + j];
                    for (k..k_end) |p| {
                        sum += lhs[i * K + p] * rhs[p * N + j];
                    }
                    out[i * N + j] = sum;
                }
            }
        }
    }
}

/// Multi-threaded integer matrix multiplication with modulo $Q = 3329$ operation.
/// Formula: $$C_{i,j} = \left( \sum_{k} A_{i,k} B_{k,j} \right) \pmod{Q}$$
pub fn mod_matmul(M: usize, K: usize, N: usize, lhs: []const i32, rhs: []const i32, out: []i32) void {
    const MC = 64;
    const num_threads = @min(getNumThreads(), max_threads);

    @memset(out, 0);

    if (num_threads <= 1 or M < MC) {
        modMatmulKernel(0, M, K, N, lhs, rhs, out);
        return;
    }

    const num_row_blocks: usize = (M + MC - 1) / MC;
    const blocks_per_thread: usize = (num_row_blocks + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const block_start: usize = t * blocks_per_thread;
        if (block_start >= num_row_blocks) break;
        const block_end: usize = @min(block_start + blocks_per_thread, num_row_blocks);
        const m_start: usize = block_start * MC;
        const m_end: usize = @min(block_end * MC, M);

        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(m_s: usize, m_e: usize, K_: usize, N_: usize, lhs_: []const i32, rhs_: []const i32, out_: []i32) void {
                modMatmulKernel(m_s, m_e, K_, N_, lhs_, rhs_, out_);
            }
        }.worker, .{ m_start, m_end, K, N, lhs, rhs, out }) catch {
            modMatmulKernel(m_start, m_end, K, N, lhs, rhs, out);
            continue;
        };
        spawned += 1;
    }

    for (threads[0..spawned]) |t| t.join();
}

/// Vectorized kernel for integer matrix multiplication with modulo $Q = 3329$ operation.
/// Formula: $$C_{i,j} = \left( \sum_{k} A_{i,k} B_{k,j} \right) \pmod{Q}$$
fn modMatmulKernel(m_start: usize, m_end: usize, K: usize, N: usize, lhs: []const i32, rhs: []const i32, out: []i32) void {
    const Q: i64 = 3329;
    const VL: usize = 16;
    const Vec: type = @Vector(VL, i32);
    const Vec64: type = @Vector(VL, i64);
    const NC: usize = 64;

    var n: usize = 0;
    while (n < N) : (n += NC) {
        const n_end = @min(N, n + NC);
        for (m_start..m_end) |i| {
            var j: usize = n;
            while (j + VL <= n_end) : (j += VL) {
                var sum_vec: Vec64 = @splat(0);
                var k: usize = 0;
                while (k < K) : (k += 1) {
                    const a_val: i64 = lhs[i * K + k];
                    const a_vec: Vec64 = @splat(a_val);
                    const b_vec: Vec = rhs[k * N + j ..][0..VL].*;
                    var b_vec64: Vec64 = undefined;
                    const b_arr: [VL]i32 = b_vec;
                    inline for (b_arr, 0..) |b, idx| b_vec64[idx] = b;
                    sum_vec += a_vec * b_vec64;
                }
                var out_arr: [VL]i32 = undefined;
                const sum_arr: [VL]i64 = sum_vec;
                for (sum_arr, 0..) |s, idx| {
                    out_arr[idx] = @intCast(@mod(s, Q));
                }
                const out_vec: Vec = out_arr;
                out[i * N + j ..][0..VL].* = out_vec;
            }
            while (j < n_end) : (j += 1) {
                var sum: i64 = 0;
                for (0..K) |k| {
                    sum += @as(i64, lhs[i * K + k]) * @as(i64, rhs[k * N + j]);
                }
                out[i * N + j] = @intCast(@mod(sum, Q));
            }
        }
    }
}

/// Multi-threaded simulation of 2D heat transfer across a grid step.
/// Formula: $$U_{i,j}^{(t+1)} = \frac{1}{4} \left( U_{i-1,j}^{(t)} + U_{i+1,j}^{(t)} + U_{i,j-1}^{(t)} + U_{i,j+1}^{(t)} \right)$$
pub fn heat_transfer(H: usize, W: usize, grid: []const f32, out: []f32) void {
    const num_threads: usize = @min(getNumThreads(), max_threads);

    // Copy everything first to preserve boundaries
    @memcpy(out, grid);

    if (num_threads <= 1 or H < 64) {
        heatTransferKernel(1, H - 1, W, grid, out);
        return;
    }

    const rows = H - 2; // Rows to process: 1..H-1
    const chunk_size: usize = (rows + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |t| {
        const r_start: usize = 1 + t * chunk_size;
        if (r_start >= H - 1) break;
        const r_end: usize = @min(r_start + chunk_size, H - 1);

        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(start_row: usize, end_row: usize, Width: usize, g: []const f32, o: []f32) void {
                heatTransferKernel(start_row, end_row, Width, g, o);
            }
        }.worker, .{ r_start, r_end, W, grid, out }) catch {
            heatTransferKernel(r_start, r_end, W, grid, out);
            continue;
        };
        spawned += 1;
    }
    for (threads[0..spawned]) |t| t.join();
}

/// Vectorized kernel for 2D heat transfer across a grid step.
/// Formula: $$U_{i,j}^{(t+1)} = \frac{1}{4} \left( U_{i-1,j}^{(t)} + U_{i+1,j}^{(t)} + U_{i,j-1}^{(t)} + U_{i,j+1}^{(t)} \right)$$
fn heatTransferKernel(row_start: usize, row_end: usize, W: usize, grid: []const f32, out: []f32) void {
    const VL: usize = 64;
    const Vec: type = @Vector(VL, f32);
    const quarter: Vec = @splat(0.25);

    for (row_start..row_end) |i| {
        var j: usize = 1;
        while (j + VL <= W - 1) : (j += VL) {
            const up: Vec = grid[(i - 1) * W + j ..][0..VL].*;
            const down: Vec = grid[(i + 1) * W + j ..][0..VL].*;
            const left: Vec = grid[i * W + j - 1 ..][0..VL].*;
            const right: Vec = grid[i * W + j + 1 ..][0..VL].*;

            out[i * W + j ..][0..VL].* = (up + down + left + right) * quarter;
        }
        while (j < W - 1) : (j += 1) {
            const up = grid[(i - 1) * W + j];
            const down = grid[(i + 1) * W + j];
            const left = grid[i * W + j - 1];
            const right = grid[i * W + j + 1];
            out[i * W + j] = 0.25 * (up + down + left + right);
        }
    }
}

/// Vectorized approximation of the standard normal cumulative distribution function.
fn stdNormalCdf(comptime VL: usize, x: @Vector(VL, f32)) @Vector(VL, f32) {
    const Vec: type = @Vector(VL, f32);
    const p: Vec = @splat(0.3275911);
    const a1: Vec = @splat(0.254829592);
    const a2: Vec = @splat(-0.284496736);
    const a3: Vec = @splat(1.421413741);
    const a4: Vec = @splat(-1.453152027);
    const a5: Vec = @splat(1.061405429);
    const one: Vec = @splat(1.0);
    const zero: Vec = @splat(0.0);
    const half: Vec = @splat(0.5);
    const sqrt2_inv: Vec = @splat(1.0 / std.math.sqrt(2.0));

    const v = x * sqrt2_inv;
    const z = @abs(v);

    const t = one / (one + p * z);
    const poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    const y = one - poly * @exp(-z * z);

    const sign_mask = v >= zero;
    const sign = @select(f32, sign_mask, one, @as(Vec, @splat(-1.0)));
    const erf = sign * y;

    return half * (one + erf);
}

/// Multi-threaded evaluation of the Black-Scholes pricing model for physical options.
/// Formulas for European call:
/// $$C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$
/// $$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}$$
/// $$d_2 = d_1 - \sigma \sqrt{T}$$
pub fn black_scholes(n: usize, s: []const f32, k: []const f32, t: []const f32, r: []const f32, sigma: []const f32, call: []f32, put: []f32) void {
    const num_threads = @min(getNumThreads(), max_threads);
    if (num_threads <= 1 or n < 1024) {
        blackScholesKernel(n, s, k, t, r, sigma, call, put);
        return;
    }

    const chunk_size = (n + num_threads - 1) / num_threads;
    var threads: [max_threads]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..num_threads) |th| {
        const start = th * chunk_size;
        if (start >= n) break;
        const end = @min(start + chunk_size, n);

        threads[spawned] = std.Thread.spawn(.{}, struct {
            fn worker(len: usize, s_: []const f32, k_: []const f32, t_: []const f32, r_: []const f32, sigma_: []const f32, call_: []f32, put_: []f32) void {
                blackScholesKernel(len, s_, k_, t_, r_, sigma_, call_, put_);
            }
        }.worker, .{ end - start, s[start..end], k[start..end], t[start..end], r[start..end], sigma[start..end], call[start..end], put[start..end] }) catch {
            blackScholesKernel(end - start, s[start..end], k[start..end], t[start..end], r[start..end], sigma[start..end], call[start..end], put[start..end]);
            continue;
        };
        spawned += 1;
    }
    for (threads[0..spawned]) |th| th.join();
}

/// Vectorized kernel for evaluating the Black-Scholes pricing model.
/// Formulas for European call:
/// $$C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$
/// $$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}$$
/// $$d_2 = d_1 - \sigma \sqrt{T}$$
fn blackScholesKernel(n: usize, s: []const f32, k: []const f32, t: []const f32, r: []const f32, sigma: []const f32, call: []f32, put: []f32) void {
    const VL: usize = 64;
    const Vec: type = @Vector(VL, f32);

    var i: usize = 0;
    while (i + VL <= n) : (i += VL) {
        const s_vec: Vec = s[i..][0..VL].*;
        const k_vec: Vec = k[i..][0..VL].*;
        const t_vec: Vec = t[i..][0..VL].*;
        const r_vec: Vec = r[i..][0..VL].*;
        const sigma_vec: Vec = sigma[i..][0..VL].*;

        const sqrt_t: Vec = @sqrt(t_vec);
        const sigma_sqrt_t: Vec = sigma_vec * sqrt_t;
        const half_sigma2: Vec = @as(Vec, @splat(0.5)) * sigma_vec * sigma_vec;
        const d1: Vec = (@log(s_vec / k_vec) + (r_vec + half_sigma2) * t_vec) / sigma_sqrt_t;
        const d2: Vec = d1 - sigma_sqrt_t;

        const cdf_d1: Vec = stdNormalCdf(VL, d1);
        const cdf_d2: Vec = stdNormalCdf(VL, d2);
        const k_exp_rt: Vec = k_vec * @exp(-r_vec * t_vec);

        call[i..][0..VL].* = s_vec * cdf_d1 - k_exp_rt * cdf_d2;

        const cdf_neg_d1: Vec = stdNormalCdf(VL, -d1);
        const cdf_neg_d2: Vec = stdNormalCdf(VL, -d2);
        put[i..][0..VL].* = k_exp_rt * cdf_neg_d2 - s_vec * cdf_neg_d1;
    }

    while (i < n) : (i += 1) {
        const sqrt_t: f32 = @sqrt(t[i]);
        const sigma_sqrt_t: f32 = sigma[i] * sqrt_t;
        const d1: f32 = (@log(s[i] / k[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * t[i]) / sigma_sqrt_t;
        const d2: f32 = d1 - sigma_sqrt_t;

        const cdf_d1: f32 = stdNormalCdf(1, @as(@Vector(1, f32), @splat(d1)))[0];
        const cdf_d2: f32 = stdNormalCdf(1, @as(@Vector(1, f32), @splat(d2)))[0];
        const k_exp_rt: f32 = k[i] * @exp(-r[i] * t[i]);

        call[i] = s[i] * cdf_d1 - k_exp_rt * cdf_d2;

        const cdf_neg_d1: f32 = stdNormalCdf(1, @as(@Vector(1, f32), @splat(-d1)))[0];
        const cdf_neg_d2: f32 = stdNormalCdf(1, @as(@Vector(1, f32), @splat(-d2)))[0];
        put[i] = k_exp_rt * cdf_neg_d2 - s[i] * cdf_neg_d1;
    }
}
