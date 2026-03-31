const std = @import("std");

/// Computes standard matrix multiplication.
/// Formula: $$C_{i,j} = \sum_{k} A_{i,k} B_{k,j}$$
pub fn matmul(M: usize, K: usize, N: usize, lhs: []const f32, rhs: []const f32, out: []f32) void {
    for (0..M) |m| {
        for (0..N) |n| {
            var sum: f32 = 0.0;
            for (0..K) |k| {
                sum += lhs[m * K + k] * rhs[k * N + n];
            }
            out[m * N + n] = sum;
        }
    }
}

/// Single-precision scalar multiplication and vector addition.
/// Formula: $$Z_i = a \cdot X_i + Y_i$$
pub fn saxpy(n: usize, a: f32, x: []const f32, y: []const f32, out: []f32) void {
    for (0..n) |i| {
        out[i] = a * x[i] + y[i];
    }
}

/// Integer matrix multiplication with modulo $Q = 3329$ operation.
/// Formula: $$C_{i,j} = \left( \sum_{k} A_{i,k} B_{k,j} \right) \pmod{Q}$$
pub fn mod_matmul(M: usize, K: usize, N: usize, lhs: []const i32, rhs: []const i32, out: []i32) void {
    const Q: i64 = 3329;
    for (0..M) |m| {
        for (0..N) |n| {
            var sum: i64 = 0;
            for (0..K) |k| {
                const a: i64 = lhs[m * K + k];
                const b: i64 = rhs[k * N + n];
                sum += a * b;
            }
            out[m * N + n] = @intCast(@mod(sum, Q));
        }
    }
}

/// Simulates 2D heat transfer across a grid step.
/// Formula: $$U_{i,j}^{(t+1)} = \frac{1}{4} \left( U_{i-1,j}^{(t)} + U_{i+1,j}^{(t)} + U_{i,j-1}^{(t)} + U_{i,j+1}^{(t)} \right)$$
pub fn heat_transfer(H: usize, W: usize, grid: []const f32, out: []f32) void {
    @memcpy(out, grid);
    for (1..H - 1) |i| {
        for (1..W - 1) |j| {
            const up = grid[(i - 1) * W + j];
            const down = grid[(i + 1) * W + j];
            const left = grid[i * W + j - 1];
            const right = grid[i * W + j + 1];
            out[i * W + j] = 0.25 * (up + down + left + right);
        }
    }
}

/// Calculates the standard normal cumulative distribution function.
/// This approximation uses an error function defined by polynomial constants.
fn std_normal_cdf(x: f32) f32 {
    const p: f32 = 0.3275911;
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;

    const v: f32 = x / @sqrt(2.0);
    const z: f32 = @abs(v);

    const t: f32 = 1.0 / (1.0 + p * z);
    const poly: f32 = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    const y: f32 = 1.0 - poly * @exp(-z * z);

    const sign: f32 = if (v >= 0) 1.0 else -1.0;
    const erf = sign * y;

    return 0.5 * (1.0 + erf);
}

/// Evaluates the Black-Scholes pricing model for physical options.
/// Formulas for European call:
/// $$C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$
/// $$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}$$
/// $$d_2 = d_1 - \sigma \sqrt{T}$$
pub fn black_scholes(n: usize, s: []const f32, k: []const f32, t: []const f32, r: []const f32, sigma: []const f32, call: []f32, put: []f32) void {
    for (0..n) |i| {
        const sqrt_t: f32 = @sqrt(t[i]);
        const sigma_sqrt_t: f32 = sigma[i] * sqrt_t;
        const d1: f32 = (@log(s[i] / k[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * t[i]) / sigma_sqrt_t;
        const d2: f32 = d1 - sigma_sqrt_t;

        const cdf_d1: f32 = std_normal_cdf(d1);
        const cdf_d2: f32 = std_normal_cdf(d2);
        const k_exp_rt: f32 = k[i] * @exp(-r[i] * t[i]);

        call[i] = s[i] * cdf_d1 - k_exp_rt * cdf_d2;

        const cdf_neg_d1: f32 = std_normal_cdf(-d1);
        const cdf_neg_d2: f32 = std_normal_cdf(-d2);
        put[i] = k_exp_rt * cdf_neg_d2 - s[i] * cdf_neg_d1;
    }
}
