const std = @import("std");
const dietz = @import("dietz");

const Tree = dietz.Tree;
const ThreadSafeTree = dietz.ThreadSafeTree;

const Range = struct { first: i64, last: i64 };
const ThreadRange = struct { start: i64, end: i64 };

// ============================================================================
// Memory Tracking Allocator
// ============================================================================

const TrackingAllocator = struct {
    backing: std.mem.Allocator,
    bytes_allocated: usize = 0,
    bytes_freed: usize = 0,
    allocation_count: usize = 0,
    free_count: usize = 0,
    peak_bytes: usize = 0,

    fn init(backing: std.mem.Allocator) TrackingAllocator {
        return .{ .backing = backing };
    }

    fn allocator(self: *TrackingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing.rawAlloc(len, alignment, ret_addr);
        if (result != null) {
            self.bytes_allocated += len;
            self.allocation_count += 1;
            const current = self.bytes_allocated - self.bytes_freed;
            self.peak_bytes = @max(self.peak_bytes, current);
        }
        return result;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        if (self.backing.rawResize(buf, alignment, new_len, ret_addr)) {
            if (new_len > buf.len) {
                self.bytes_allocated += new_len - buf.len;
            } else {
                self.bytes_freed += buf.len - new_len;
            }
            const current = self.bytes_allocated - self.bytes_freed;
            self.peak_bytes = @max(self.peak_bytes, current);
            return true;
        }
        return false;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = buf;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        self.bytes_freed += buf.len;
        self.free_count += 1;
        self.backing.rawFree(buf, alignment, ret_addr);
    }

    fn reset(self: *TrackingAllocator) void {
        self.bytes_allocated = 0;
        self.bytes_freed = 0;
        self.allocation_count = 0;
        self.free_count = 0;
        self.peak_bytes = 0;
    }
};

// ============================================================================
// Configuration
// ============================================================================

const Config = struct {
    iterations: usize = 100,
    warmup: usize = 5,
    sizes: SizeSet = .{ .small = true, .medium = true, .large = true },
    format: Format = .text,
    filter: ?[]const u8 = null,
    thread_count: usize = 0, // 0 = auto-detect

    const SizeSet = struct {
        small: bool = false,
        medium: bool = false,
        large: bool = false,
    };

    const Format = enum { text, json };
};

const Size = enum {
    small,
    medium,
    large,

    fn elementCount(self: Size) usize {
        return switch (self) {
            .small => 100,
            .medium => 10_000,
            .large => 1_000_000,
        };
    }

    fn name(self: Size) []const u8 {
        return switch (self) {
            .small => "S",
            .medium => "M",
            .large => "L",
        };
    }
};

// ============================================================================
// Statistics
// ============================================================================

const Stats = struct {
    samples: std.ArrayList(u64),
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) Stats {
        return .{
            .samples = .empty,
            .allocator = allocator,
        };
    }

    fn deinit(self: *Stats) void {
        self.samples.deinit(self.allocator);
    }

    fn add(self: *Stats, sample: u64) !void {
        try self.samples.append(self.allocator, sample);
    }

    fn min(self: *const Stats) u64 {
        if (self.samples.items.len == 0) return 0;
        var result: u64 = std.math.maxInt(u64);
        for (self.samples.items) |s| {
            result = @min(result, s);
        }
        return result;
    }

    fn max(self: *const Stats) u64 {
        if (self.samples.items.len == 0) return 0;
        var result: u64 = 0;
        for (self.samples.items) |s| {
            result = @max(result, s);
        }
        return result;
    }

    fn mean(self: *const Stats) u64 {
        if (self.samples.items.len == 0) return 0;
        var sum: u128 = 0;
        for (self.samples.items) |s| {
            sum += s;
        }
        return @intCast(sum / self.samples.items.len);
    }

    fn median(self: *Stats) u64 {
        if (self.samples.items.len == 0) return 0;
        std.mem.sort(u64, self.samples.items, {}, std.sort.asc(u64));
        const mid = self.samples.items.len / 2;
        if (self.samples.items.len % 2 == 0) {
            return (self.samples.items[mid - 1] + self.samples.items[mid]) / 2;
        }
        return self.samples.items[mid];
    }

    fn stddev(self: *const Stats) u64 {
        if (self.samples.items.len < 2) return 0;
        const avg = self.mean();
        var sum_sq: u128 = 0;
        for (self.samples.items) |s| {
            const diff: i128 = @as(i128, s) - @as(i128, avg);
            sum_sq += @intCast(@as(u128, @intCast(diff * diff)));
        }
        const variance = sum_sq / (self.samples.items.len - 1);
        return std.math.sqrt(variance);
    }
};

// ============================================================================
// Benchmark Result
// ============================================================================

const BenchmarkResult = struct {
    name: []const u8,
    size: Size,
    stats: struct {
        mean_ns: u64,
        std_ns: u64,
        min_ns: u64,
        max_ns: u64,
        median_ns: u64,
    },
    memory: struct {
        peak_bytes: usize,
        allocation_count: usize,
    },
};

// ============================================================================
// Output Formatters
// ============================================================================

fn formatNanos(ns: u64) struct { value: f64, unit: []const u8 } {
    if (ns >= 1_000_000_000) {
        return .{ .value = @as(f64, @floatFromInt(ns)) / 1_000_000_000.0, .unit = "s" };
    } else if (ns >= 1_000_000) {
        return .{ .value = @as(f64, @floatFromInt(ns)) / 1_000_000.0, .unit = "ms" };
    } else if (ns >= 1_000) {
        return .{ .value = @as(f64, @floatFromInt(ns)) / 1_000.0, .unit = "us" };
    } else {
        return .{ .value = @as(f64, @floatFromInt(ns)), .unit = "ns" };
    }
}

fn formatBytes(bytes: usize) struct { value: f64, unit: []const u8 } {
    if (bytes >= 1024 * 1024 * 1024) {
        return .{ .value = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0), .unit = "GB" };
    } else if (bytes >= 1024 * 1024) {
        return .{ .value = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0), .unit = "MB" };
    } else if (bytes >= 1024) {
        return .{ .value = @as(f64, @floatFromInt(bytes)) / 1024.0, .unit = "KB" };
    } else {
        return .{ .value = @as(f64, @floatFromInt(bytes)), .unit = "B" };
    }
}

fn formatOpsPerSec(ns: u64) struct { value: f64, unit: []const u8 } {
    if (ns == 0) return .{ .value = 0, .unit = "ops/s" };
    const ops_per_sec = 1_000_000_000.0 / @as(f64, @floatFromInt(ns));
    if (ops_per_sec >= 1_000_000_000) {
        return .{ .value = ops_per_sec / 1_000_000_000.0, .unit = "Gop/s" };
    } else if (ops_per_sec >= 1_000_000) {
        return .{ .value = ops_per_sec / 1_000_000.0, .unit = "Mop/s" };
    } else if (ops_per_sec >= 1_000) {
        return .{ .value = ops_per_sec / 1_000.0, .unit = "Kop/s" };
    } else {
        return .{ .value = ops_per_sec, .unit = "op/s" };
    }
}

fn printTextResult(result: BenchmarkResult, buf: []u8) void {
    const m = formatNanos(result.stats.mean_ns);
    const s = formatNanos(result.stats.std_ns);
    const mem = formatBytes(result.memory.peak_bytes);
    const ops = formatOpsPerSec(result.stats.mean_ns);

    const line = std.fmt.bufPrint(buf, "  {s:<28} {d:>7.3} {s:<2}  ±{d:>7.3} {s:<2}  {d:>7.2} {s:<5}  peak: {d:>6.1} {s:<2}  ({d} allocs)\n", .{
        result.name,
        m.value,
        m.unit,
        s.value,
        s.unit,
        ops.value,
        ops.unit,
        mem.value,
        mem.unit,
        result.memory.allocation_count,
    }) catch return;
    std.fs.File.stdout().writeAll(line) catch {};
}

fn printJsonResults(results: []const BenchmarkResult, config: Config, buf: []u8) void {
    const stdout = std.fs.File.stdout();

    stdout.writeAll("{\n") catch return;
    const cfg_line = std.fmt.bufPrint(buf, "  \"config\": {{\"iterations\": {d}, \"warmup\": {d}}},\n", .{ config.iterations, config.warmup }) catch return;
    stdout.writeAll(cfg_line) catch return;
    stdout.writeAll("  \"benchmarks\": [\n") catch return;

    for (results, 0..) |result, i| {
        stdout.writeAll("    {\n") catch return;
        const name_line = std.fmt.bufPrint(buf, "      \"name\": \"{s}\",\n", .{result.name}) catch return;
        stdout.writeAll(name_line) catch return;
        const size_line = std.fmt.bufPrint(buf, "      \"size\": \"{s}\",\n", .{result.size.name()}) catch return;
        stdout.writeAll(size_line) catch return;
        stdout.writeAll("      \"stats\": {\n") catch return;
        const mean_line = std.fmt.bufPrint(buf, "        \"mean_ns\": {d},\n", .{result.stats.mean_ns}) catch return;
        stdout.writeAll(mean_line) catch return;
        const std_line = std.fmt.bufPrint(buf, "        \"std_ns\": {d},\n", .{result.stats.std_ns}) catch return;
        stdout.writeAll(std_line) catch return;
        const min_line = std.fmt.bufPrint(buf, "        \"min_ns\": {d},\n", .{result.stats.min_ns}) catch return;
        stdout.writeAll(min_line) catch return;
        const max_line = std.fmt.bufPrint(buf, "        \"max_ns\": {d},\n", .{result.stats.max_ns}) catch return;
        stdout.writeAll(max_line) catch return;
        const median_line = std.fmt.bufPrint(buf, "        \"median_ns\": {d}\n", .{result.stats.median_ns}) catch return;
        stdout.writeAll(median_line) catch return;
        stdout.writeAll("      },\n") catch return;
        stdout.writeAll("      \"memory\": {\n") catch return;
        const peak_line = std.fmt.bufPrint(buf, "        \"peak_bytes\": {d},\n", .{result.memory.peak_bytes}) catch return;
        stdout.writeAll(peak_line) catch return;
        const alloc_line = std.fmt.bufPrint(buf, "        \"allocation_count\": {d}\n", .{result.memory.allocation_count}) catch return;
        stdout.writeAll(alloc_line) catch return;
        stdout.writeAll("      }\n") catch return;
        if (i < results.len - 1) {
            stdout.writeAll("    },\n") catch return;
        } else {
            stdout.writeAll("    }\n") catch return;
        }
    }

    stdout.writeAll("  ]\n") catch return;
    stdout.writeAll("}\n") catch return;
}

// ============================================================================
// Benchmark Framework
// ============================================================================

fn BenchmarkFn(comptime Context: type) type {
    return *const fn (*Context) void;
}

fn SetupFn(comptime Context: type) type {
    return *const fn (std.mem.Allocator, Size) anyerror!Context;
}

fn TeardownFn(comptime Context: type) type {
    return *const fn (*Context) void;
}

fn runBenchmark(
    comptime Context: type,
    name: []const u8,
    size: Size,
    config: Config,
    allocator: std.mem.Allocator,
    setup: SetupFn(Context),
    benchmark: BenchmarkFn(Context),
    teardown: TeardownFn(Context),
) !BenchmarkResult {
    var stats = Stats.init(allocator);
    defer stats.deinit();

    var tracker = TrackingAllocator.init(allocator);
    const tracked_alloc = tracker.allocator();

    // Warmup (not tracked)
    for (0..config.warmup) |_| {
        var ctx = try setup(allocator, size);
        benchmark(&ctx);
        teardown(&ctx);
    }

    // Timed runs with memory tracking
    var max_peak: usize = 0;
    var max_allocs: usize = 0;

    for (0..config.iterations) |_| {
        tracker.reset();
        var ctx = try setup(tracked_alloc, size);

        var timer = std.time.Timer.start() catch unreachable;
        benchmark(&ctx);
        const elapsed = timer.read();

        // Capture memory stats before teardown
        max_peak = @max(max_peak, tracker.peak_bytes);
        max_allocs = @max(max_allocs, tracker.allocation_count);

        teardown(&ctx);
        try stats.add(elapsed);
    }

    return .{
        .name = name,
        .size = size,
        .stats = .{
            .mean_ns = stats.mean(),
            .std_ns = stats.stddev(),
            .min_ns = stats.min(),
            .max_ns = stats.max(),
            .median_ns = stats.median(),
        },
        .memory = .{
            .peak_bytes = max_peak,
            .allocation_count = max_allocs,
        },
    };
}

// ============================================================================
// PRNG for reproducible random data
// ============================================================================

const Prng = struct {
    state: u64,

    fn init(seed: u64) Prng {
        return .{ .state = seed };
    }

    fn next(self: *Prng) u64 {
        // xorshift64
        var x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        return x;
    }

    fn nextBounded(self: *Prng, bound: u64) u64 {
        return self.next() % bound;
    }

    fn shuffle(self: *Prng, comptime T: type, items: []T) void {
        if (items.len < 2) return;
        var i = items.len - 1;
        while (i > 0) : (i -= 1) {
            const j = self.nextBounded(i + 1);
            const tmp = items[i];
            items[i] = items[@intCast(j)];
            items[@intCast(j)] = tmp;
        }
    }
};

// ============================================================================
// Core Operation Benchmarks
// ============================================================================

const InsertSequentialCtx = struct {
    tree: Tree(i64),
    count: usize,

    fn setup(allocator: std.mem.Allocator, size: Size) !InsertSequentialCtx {
        return .{
            .tree = Tree(i64).init(allocator),
            .count = size.elementCount(),
        };
    }

    fn run(ctx: *InsertSequentialCtx) void {
        for (0..ctx.count) |i| {
            ctx.tree.insert(@intCast(i)) catch unreachable;
        }
    }

    fn teardown(ctx: *InsertSequentialCtx) void {
        ctx.tree.deinit();
    }
};

const InsertRandomCtx = struct {
    tree: Tree(i64),
    elements: []i64,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !InsertRandomCtx {
        const count = size.elementCount();
        const elements = try allocator.alloc(i64, count);
        for (0..count) |i| {
            elements[i] = @intCast(i);
        }
        var prng = Prng.init(12345);
        prng.shuffle(i64, elements);

        return .{
            .tree = Tree(i64).init(allocator),
            .elements = elements,
            .allocator = allocator,
        };
    }

    fn run(ctx: *InsertRandomCtx) void {
        for (ctx.elements) |elem| {
            ctx.tree.insert(elem) catch unreachable;
        }
    }

    fn teardown(ctx: *InsertRandomCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.elements);
    }
};

const InsertReverseCtx = struct {
    tree: Tree(i64),
    count: usize,

    fn setup(allocator: std.mem.Allocator, size: Size) !InsertReverseCtx {
        return .{
            .tree = Tree(i64).init(allocator),
            .count = size.elementCount(),
        };
    }

    fn run(ctx: *InsertReverseCtx) void {
        var i = ctx.count;
        while (i > 0) {
            i -= 1;
            ctx.tree.insert(@intCast(i)) catch unreachable;
        }
    }

    fn teardown(ctx: *InsertReverseCtx) void {
        ctx.tree.deinit();
    }
};

const InsertSparseCtx = struct {
    tree: Tree(i64),
    count: usize,

    fn setup(allocator: std.mem.Allocator, size: Size) !InsertSparseCtx {
        return .{
            .tree = Tree(i64).init(allocator),
            .count = size.elementCount(),
        };
    }

    fn run(ctx: *InsertSparseCtx) void {
        for (0..ctx.count) |i| {
            ctx.tree.insert(@intCast(i * 10)) catch unreachable;
        }
    }

    fn teardown(ctx: *InsertSparseCtx) void {
        ctx.tree.deinit();
    }
};

const ContainsHitCtx = struct {
    tree: Tree(i64),
    queries: []i64,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !ContainsHitCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        // Insert elements
        for (0..count) |i| {
            try tree.insert(@intCast(i));
        }

        // Create query list (shuffled existing elements)
        const queries = try allocator.alloc(i64, count);
        for (0..count) |i| {
            queries[i] = @intCast(i);
        }
        var prng = Prng.init(54321);
        prng.shuffle(i64, queries);

        return .{
            .tree = tree,
            .queries = queries,
            .allocator = allocator,
        };
    }

    fn run(ctx: *ContainsHitCtx) void {
        for (ctx.queries) |q| {
            _ = ctx.tree.contains(q);
        }
    }

    fn teardown(ctx: *ContainsHitCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.queries);
    }
};

const ContainsMissCtx = struct {
    tree: Tree(i64),
    queries: []i64,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !ContainsMissCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        // Insert sparse elements (every 10th)
        for (0..count) |i| {
            try tree.insert(@intCast(i * 10));
        }

        // Query for elements that don't exist (offset by 5)
        const queries = try allocator.alloc(i64, count);
        for (0..count) |i| {
            queries[i] = @as(i64, @intCast(i * 10)) + 5;
        }
        var prng = Prng.init(67890);
        prng.shuffle(i64, queries);

        return .{
            .tree = tree,
            .queries = queries,
            .allocator = allocator,
        };
    }

    fn run(ctx: *ContainsMissCtx) void {
        for (ctx.queries) |q| {
            _ = ctx.tree.contains(q);
        }
    }

    fn teardown(ctx: *ContainsMissCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.queries);
    }
};

const DeleteSequentialCtx = struct {
    tree: Tree(i64),
    count: usize,

    fn setup(allocator: std.mem.Allocator, size: Size) !DeleteSequentialCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);
        for (0..count) |i| {
            try tree.insert(@intCast(i));
        }
        return .{ .tree = tree, .count = count };
    }

    fn run(ctx: *DeleteSequentialCtx) void {
        for (0..ctx.count) |i| {
            ctx.tree.delete(@intCast(i));
        }
    }

    fn teardown(ctx: *DeleteSequentialCtx) void {
        ctx.tree.deinit();
    }
};

const DeleteRandomCtx = struct {
    tree: Tree(i64),
    elements: []i64,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !DeleteRandomCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);
        const elements = try allocator.alloc(i64, count);

        for (0..count) |i| {
            try tree.insert(@intCast(i));
            elements[i] = @intCast(i);
        }
        var prng = Prng.init(11111);
        prng.shuffle(i64, elements);

        return .{
            .tree = tree,
            .elements = elements,
            .allocator = allocator,
        };
    }

    fn run(ctx: *DeleteRandomCtx) void {
        for (ctx.elements) |elem| {
            ctx.tree.delete(elem);
        }
    }

    fn teardown(ctx: *DeleteRandomCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.elements);
    }
};

const DeleteSplitCtx = struct {
    tree: Tree(i64),
    midpoints: []i64,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !DeleteSplitCtx {
        const count = size.elementCount();
        const interval_size: usize = 100;
        const num_intervals = count / interval_size;

        var tree = Tree(i64).init(allocator);

        // Create intervals of size 100
        for (0..num_intervals) |i| {
            const start: i64 = @intCast(i * interval_size * 2); // Leave gaps
            for (0..interval_size) |j| {
                try tree.insert(start + @as(i64, @intCast(j)));
            }
        }

        // Midpoints to delete (forces splits)
        const midpoints = try allocator.alloc(i64, num_intervals);
        for (0..num_intervals) |i| {
            midpoints[i] = @as(i64, @intCast(i * interval_size * 2)) + @as(i64, @intCast(interval_size / 2));
        }

        return .{
            .tree = tree,
            .midpoints = midpoints,
            .allocator = allocator,
        };
    }

    fn run(ctx: *DeleteSplitCtx) void {
        for (ctx.midpoints) |mid| {
            ctx.tree.delete(mid);
        }
    }

    fn teardown(ctx: *DeleteSplitCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.midpoints);
    }
};

// ============================================================================
// Range Operation Benchmarks
// ============================================================================

const InsertRangeSmallCtx = struct {
    tree: Tree(i64),
    count: usize,

    fn setup(allocator: std.mem.Allocator, size: Size) !InsertRangeSmallCtx {
        return .{
            .tree = Tree(i64).init(allocator),
            .count = size.elementCount() / 10, // Number of ranges
        };
    }

    fn run(ctx: *InsertRangeSmallCtx) void {
        for (0..ctx.count) |i| {
            const start: i64 = @intCast(i * 20); // Ranges with gaps
            ctx.tree.insertRange(start, start + 9) catch unreachable;
        }
    }

    fn teardown(ctx: *InsertRangeSmallCtx) void {
        ctx.tree.deinit();
    }
};

const InsertRangeLargeCtx = struct {
    tree: Tree(i64),
    count: usize,

    fn setup(allocator: std.mem.Allocator, size: Size) !InsertRangeLargeCtx {
        return .{
            .tree = Tree(i64).init(allocator),
            .count = @max(1, size.elementCount() / 1000), // Number of large ranges
        };
    }

    fn run(ctx: *InsertRangeLargeCtx) void {
        for (0..ctx.count) |i| {
            const start: i64 = @intCast(i * 2000);
            ctx.tree.insertRange(start, start + 999) catch unreachable;
        }
    }

    fn teardown(ctx: *InsertRangeLargeCtx) void {
        ctx.tree.deinit();
    }
};

const ContainsRangeHitCtx = struct {
    tree: Tree(i64),
    ranges: []Range,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !ContainsRangeHitCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        // Insert contiguous elements
        for (0..count) |i| {
            try tree.insert(@intCast(i));
        }

        // Create ranges that exist within the tree
        const num_queries = @max(1, count / 100);
        const ranges = try allocator.alloc(Range, num_queries);
        var prng = Prng.init(22222);
        for (0..num_queries) |i| {
            const start: i64 = @intCast(prng.nextBounded(count - 10));
            ranges[i] = .{ .first = start, .last = start + 9 };
        }

        return .{
            .tree = tree,
            .ranges = ranges,
            .allocator = allocator,
        };
    }

    fn run(ctx: *ContainsRangeHitCtx) void {
        for (ctx.ranges) |r| {
            _ = ctx.tree.containsRange(r.first, r.last);
        }
    }

    fn teardown(ctx: *ContainsRangeHitCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.ranges);
    }
};

const ContainsRangeMissCtx = struct {
    tree: Tree(i64),
    ranges: []Range,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !ContainsRangeMissCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        // Insert sparse elements
        for (0..count) |i| {
            try tree.insert(@intCast(i * 10));
        }

        // Create ranges that span gaps
        const num_queries = @max(1, count / 100);
        const ranges = try allocator.alloc(Range, num_queries);
        for (0..num_queries) |i| {
            const start: i64 = @as(i64, @intCast(i * 10)) + 3;
            ranges[i] = .{ .first = start, .last = start + 5 }; // Spans gap
        }

        return .{
            .tree = tree,
            .ranges = ranges,
            .allocator = allocator,
        };
    }

    fn run(ctx: *ContainsRangeMissCtx) void {
        for (ctx.ranges) |r| {
            _ = ctx.tree.containsRange(r.first, r.last);
        }
    }

    fn teardown(ctx: *ContainsRangeMissCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.ranges);
    }
};

const IntersectsCtx = struct {
    tree: Tree(i64),
    ranges: []Range,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !IntersectsCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        // Insert sparse elements
        for (0..count) |i| {
            try tree.insert(@intCast(i * 5));
        }

        // Create ranges that partially overlap
        const num_queries = @max(1, count / 100);
        const ranges = try allocator.alloc(Range, num_queries);
        var prng = Prng.init(33333);
        for (0..num_queries) |i| {
            const start: i64 = @intCast(prng.nextBounded(count * 5));
            ranges[i] = .{ .first = start, .last = start + 10 };
        }

        return .{
            .tree = tree,
            .ranges = ranges,
            .allocator = allocator,
        };
    }

    fn run(ctx: *IntersectsCtx) void {
        for (ctx.ranges) |r| {
            _ = ctx.tree.intersects(r.first, r.last);
        }
    }

    fn teardown(ctx: *IntersectsCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.ranges);
    }
};

const DeleteRangeCtx = struct {
    tree: Tree(i64),
    ranges: []Range,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !DeleteRangeCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        for (0..count) |i| {
            try tree.insert(@intCast(i));
        }

        // Create ranges to delete
        const num_ranges = @max(1, count / 100);
        const ranges = try allocator.alloc(Range, num_ranges);
        for (0..num_ranges) |i| {
            const start: i64 = @intCast(i * 100);
            ranges[i] = .{ .first = start, .last = start + 9 };
        }

        return .{
            .tree = tree,
            .ranges = ranges,
            .allocator = allocator,
        };
    }

    fn run(ctx: *DeleteRangeCtx) void {
        for (ctx.ranges) |r| {
            ctx.tree.deleteRange(r.first, r.last);
        }
    }

    fn teardown(ctx: *DeleteRangeCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.ranges);
    }
};

// ============================================================================
// Set Operation Benchmarks
// ============================================================================

const SetUnionDisjointCtx = struct {
    tree1: Tree(i64),
    tree2: Tree(i64),
    result: ?Tree(i64),
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !SetUnionDisjointCtx {
        const count = size.elementCount();
        var tree1 = Tree(i64).init(allocator);
        var tree2 = Tree(i64).init(allocator);

        // Tree1: 0..count
        for (0..count) |i| {
            try tree1.insert(@intCast(i));
        }
        // Tree2: count*2..count*3 (disjoint)
        for (0..count) |i| {
            try tree2.insert(@as(i64, @intCast(count * 2)) + @as(i64, @intCast(i)));
        }

        return .{
            .tree1 = tree1,
            .tree2 = tree2,
            .result = null,
            .allocator = allocator,
        };
    }

    fn run(ctx: *SetUnionDisjointCtx) void {
        ctx.result = ctx.tree1.setUnion(&ctx.tree2, ctx.allocator) catch null;
    }

    fn teardown(ctx: *SetUnionDisjointCtx) void {
        if (ctx.result) |*r| r.deinit();
        ctx.tree1.deinit();
        ctx.tree2.deinit();
    }
};

const SetUnionOverlappingCtx = struct {
    tree1: Tree(i64),
    tree2: Tree(i64),
    result: ?Tree(i64),
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !SetUnionOverlappingCtx {
        const count = size.elementCount();
        var tree1 = Tree(i64).init(allocator);
        var tree2 = Tree(i64).init(allocator);

        // Tree1: 0..count
        for (0..count) |i| {
            try tree1.insert(@intCast(i));
        }
        // Tree2: count/2..count*3/2 (50% overlap)
        for (0..count) |i| {
            try tree2.insert(@as(i64, @intCast(count / 2)) + @as(i64, @intCast(i)));
        }

        return .{
            .tree1 = tree1,
            .tree2 = tree2,
            .result = null,
            .allocator = allocator,
        };
    }

    fn run(ctx: *SetUnionOverlappingCtx) void {
        ctx.result = ctx.tree1.setUnion(&ctx.tree2, ctx.allocator) catch null;
    }

    fn teardown(ctx: *SetUnionOverlappingCtx) void {
        if (ctx.result) |*r| r.deinit();
        ctx.tree1.deinit();
        ctx.tree2.deinit();
    }
};

const SetIntersectionCtx = struct {
    tree1: Tree(i64),
    tree2: Tree(i64),
    result: ?Tree(i64),
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !SetIntersectionCtx {
        const count = size.elementCount();
        var tree1 = Tree(i64).init(allocator);
        var tree2 = Tree(i64).init(allocator);

        for (0..count) |i| {
            try tree1.insert(@intCast(i));
        }
        for (0..count) |i| {
            try tree2.insert(@as(i64, @intCast(count / 2)) + @as(i64, @intCast(i)));
        }

        return .{
            .tree1 = tree1,
            .tree2 = tree2,
            .result = null,
            .allocator = allocator,
        };
    }

    fn run(ctx: *SetIntersectionCtx) void {
        ctx.result = ctx.tree1.setIntersection(&ctx.tree2, ctx.allocator) catch null;
    }

    fn teardown(ctx: *SetIntersectionCtx) void {
        if (ctx.result) |*r| r.deinit();
        ctx.tree1.deinit();
        ctx.tree2.deinit();
    }
};

const SetDifferenceCtx = struct {
    tree1: Tree(i64),
    tree2: Tree(i64),
    result: ?Tree(i64),
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !SetDifferenceCtx {
        const count = size.elementCount();
        var tree1 = Tree(i64).init(allocator);
        var tree2 = Tree(i64).init(allocator);

        for (0..count) |i| {
            try tree1.insert(@intCast(i));
        }
        for (0..count) |i| {
            try tree2.insert(@as(i64, @intCast(count / 2)) + @as(i64, @intCast(i)));
        }

        return .{
            .tree1 = tree1,
            .tree2 = tree2,
            .result = null,
            .allocator = allocator,
        };
    }

    fn run(ctx: *SetDifferenceCtx) void {
        ctx.result = ctx.tree1.setDifference(&ctx.tree2, ctx.allocator) catch null;
    }

    fn teardown(ctx: *SetDifferenceCtx) void {
        if (ctx.result) |*r| r.deinit();
        ctx.tree1.deinit();
        ctx.tree2.deinit();
    }
};

// ============================================================================
// Traversal Benchmarks
// ============================================================================

const IteratorCtx = struct {
    tree: Tree(i64),

    fn setup(allocator: std.mem.Allocator, size: Size) !IteratorCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        // Create many intervals (sparse insertion)
        for (0..count) |i| {
            try tree.insert(@intCast(i * 3));
        }

        return .{ .tree = tree };
    }

    fn run(ctx: *IteratorCtx) void {
        var it = ctx.tree.iterator();
        var sum: i64 = 0;
        while (it.next()) |interval| {
            sum += interval.first;
        }
        std.mem.doNotOptimizeAway(sum);
    }

    fn teardown(ctx: *IteratorCtx) void {
        ctx.tree.deinit();
    }
};

const IntervalsCtx = struct {
    tree: Tree(i64),
    result: ?[]Tree(i64).IntervalType,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !IntervalsCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        for (0..count) |i| {
            try tree.insert(@intCast(i * 3));
        }

        return .{
            .tree = tree,
            .result = null,
            .allocator = allocator,
        };
    }

    fn run(ctx: *IntervalsCtx) void {
        ctx.result = ctx.tree.intervals(ctx.allocator) catch null;
    }

    fn teardown(ctx: *IntervalsCtx) void {
        if (ctx.result) |r| ctx.allocator.free(r);
        ctx.tree.deinit();
    }
};

const CountCtx = struct {
    tree: Tree(i64),

    fn setup(allocator: std.mem.Allocator, size: Size) !CountCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        for (0..count) |i| {
            try tree.insert(@intCast(i));
        }

        return .{ .tree = tree };
    }

    fn run(ctx: *CountCtx) void {
        const c = ctx.tree.count();
        std.mem.doNotOptimizeAway(c);
    }

    fn teardown(ctx: *CountCtx) void {
        ctx.tree.deinit();
    }
};

// ============================================================================
// Concurrency Benchmarks
// ============================================================================

const ConcurrentReadCtx = struct {
    tree: ThreadSafeTree(i64),
    queries: []i64,
    thread_count: usize,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !ConcurrentReadCtx {
        const count = size.elementCount();
        var tree = ThreadSafeTree(i64).init(allocator);

        for (0..count) |i| {
            try tree.insert(@intCast(i));
        }

        const queries = try allocator.alloc(i64, count);
        for (0..count) |i| {
            queries[i] = @intCast(i);
        }
        var prng = Prng.init(44444);
        prng.shuffle(i64, queries);

        return .{
            .tree = tree,
            .queries = queries,
            .thread_count = @max(2, std.Thread.getCpuCount() catch 4),
            .allocator = allocator,
        };
    }

    fn run(ctx: *ConcurrentReadCtx) void {
        const queries_per_thread = ctx.queries.len / ctx.thread_count;

        const ThreadContext = struct {
            tree: *ThreadSafeTree(i64),
            queries: []const i64,
        };

        const worker = struct {
            fn work(tc: ThreadContext) void {
                for (tc.queries) |q| {
                    _ = tc.tree.contains(q);
                }
            }
        }.work;

        var threads: [32]std.Thread = undefined;
        const actual_threads = @min(ctx.thread_count, 32);

        for (0..actual_threads) |i| {
            const start = i * queries_per_thread;
            const end = if (i == actual_threads - 1) ctx.queries.len else (i + 1) * queries_per_thread;
            threads[i] = std.Thread.spawn(.{}, worker, .{ThreadContext{
                .tree = &ctx.tree,
                .queries = ctx.queries[start..end],
            }}) catch unreachable;
        }

        for (0..actual_threads) |i| {
            threads[i].join();
        }
    }

    fn teardown(ctx: *ConcurrentReadCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.queries);
    }
};

const ConcurrentWriteCtx = struct {
    tree: ThreadSafeTree(i64),
    ranges: []ThreadRange,
    thread_count: usize,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !ConcurrentWriteCtx {
        const count = size.elementCount();
        const thread_count = @max(2, std.Thread.getCpuCount() catch 4);
        const per_thread = count / thread_count;

        const ranges = try allocator.alloc(ThreadRange, thread_count);
        for (0..thread_count) |i| {
            ranges[i] = .{
                .start = @intCast(i * per_thread),
                .end = @intCast((i + 1) * per_thread),
            };
        }

        return .{
            .tree = ThreadSafeTree(i64).init(allocator),
            .ranges = ranges,
            .thread_count = thread_count,
            .allocator = allocator,
        };
    }

    fn run(ctx: *ConcurrentWriteCtx) void {
        const ThreadContext = struct {
            tree: *ThreadSafeTree(i64),
            start: i64,
            end: i64,
        };

        const worker = struct {
            fn work(tc: ThreadContext) void {
                var i = tc.start;
                while (i < tc.end) : (i += 1) {
                    tc.tree.insert(i) catch {};
                }
            }
        }.work;

        var threads: [32]std.Thread = undefined;
        const actual_threads = @min(ctx.thread_count, 32);

        for (0..actual_threads) |i| {
            threads[i] = std.Thread.spawn(.{}, worker, .{ThreadContext{
                .tree = &ctx.tree,
                .start = ctx.ranges[i].start,
                .end = ctx.ranges[i].end,
            }}) catch unreachable;
        }

        for (0..actual_threads) |i| {
            threads[i].join();
        }
    }

    fn teardown(ctx: *ConcurrentWriteCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.ranges);
    }
};

const ConcurrentMixed90_10Ctx = struct {
    tree: ThreadSafeTree(i64),
    read_queries: []i64,
    write_values: []i64,
    thread_count: usize,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !ConcurrentMixed90_10Ctx {
        const count = size.elementCount();
        var tree = ThreadSafeTree(i64).init(allocator);

        // Pre-populate tree
        for (0..count) |i| {
            try tree.insert(@intCast(i));
        }

        const read_count = (count * 9) / 10;
        const write_count = count / 10;

        const read_queries = try allocator.alloc(i64, read_count);
        var prng = Prng.init(55555);
        for (0..read_count) |i| {
            read_queries[i] = @intCast(prng.nextBounded(count));
        }

        const write_values = try allocator.alloc(i64, write_count);
        for (0..write_count) |i| {
            write_values[i] = @as(i64, @intCast(count)) + @as(i64, @intCast(i));
        }

        return .{
            .tree = tree,
            .read_queries = read_queries,
            .write_values = write_values,
            .thread_count = @max(2, std.Thread.getCpuCount() catch 4),
            .allocator = allocator,
        };
    }

    fn run(ctx: *ConcurrentMixed90_10Ctx) void {
        const reader_threads = (ctx.thread_count * 9) / 10;
        const writer_threads = ctx.thread_count - reader_threads;

        const ReaderContext = struct {
            tree: *ThreadSafeTree(i64),
            queries: []const i64,
        };

        const WriterContext = struct {
            tree: *ThreadSafeTree(i64),
            values: []const i64,
        };

        const reader = struct {
            fn work(rc: ReaderContext) void {
                for (rc.queries) |q| {
                    _ = rc.tree.contains(q);
                }
            }
        }.work;

        const writer = struct {
            fn work(wc: WriterContext) void {
                for (wc.values) |v| {
                    wc.tree.insert(v) catch {};
                }
            }
        }.work;

        var threads: [32]std.Thread = undefined;
        var thread_idx: usize = 0;

        // Start readers
        const queries_per_reader = ctx.read_queries.len / @max(1, reader_threads);
        for (0..reader_threads) |i| {
            const start = i * queries_per_reader;
            const end = if (i == reader_threads - 1) ctx.read_queries.len else (i + 1) * queries_per_reader;
            threads[thread_idx] = std.Thread.spawn(.{}, reader, .{ReaderContext{
                .tree = &ctx.tree,
                .queries = ctx.read_queries[start..end],
            }}) catch unreachable;
            thread_idx += 1;
        }

        // Start writers
        const values_per_writer = ctx.write_values.len / @max(1, writer_threads);
        for (0..writer_threads) |i| {
            const start = i * values_per_writer;
            const end = if (i == writer_threads - 1) ctx.write_values.len else (i + 1) * values_per_writer;
            threads[thread_idx] = std.Thread.spawn(.{}, writer, .{WriterContext{
                .tree = &ctx.tree,
                .values = ctx.write_values[start..end],
            }}) catch unreachable;
            thread_idx += 1;
        }

        for (0..thread_idx) |i| {
            threads[i].join();
        }
    }

    fn teardown(ctx: *ConcurrentMixed90_10Ctx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.read_queries);
        ctx.allocator.free(ctx.write_values);
    }
};

const OverheadCtx = struct {
    regular_tree: Tree(i64),
    threadsafe_tree: *ThreadSafeTree(i64),
    queries: []i64,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !OverheadCtx {
        const count = size.elementCount();
        var regular = Tree(i64).init(allocator);

        const threadsafe = try allocator.create(ThreadSafeTree(i64));
        threadsafe.* = ThreadSafeTree(i64).init(allocator);

        for (0..count) |i| {
            try regular.insert(@intCast(i));
            try threadsafe.insert(@intCast(i));
        }

        const queries = try allocator.alloc(i64, count);
        for (0..count) |i| {
            queries[i] = @intCast(i);
        }
        var prng = Prng.init(66666);
        prng.shuffle(i64, queries);

        return .{
            .regular_tree = regular,
            .threadsafe_tree = threadsafe,
            .queries = queries,
            .allocator = allocator,
        };
    }

    fn run(ctx: *OverheadCtx) void {
        for (ctx.queries) |q| {
            _ = ctx.threadsafe_tree.contains(q);
        }
    }

    fn teardown(ctx: *OverheadCtx) void {
        ctx.regular_tree.deinit();
        ctx.threadsafe_tree.deinit();
        ctx.allocator.destroy(ctx.threadsafe_tree);
        ctx.allocator.free(ctx.queries);
    }
};

// For measuring the non-threadsafe baseline
const OverheadBaselineCtx = struct {
    tree: Tree(i64),
    queries: []i64,
    allocator: std.mem.Allocator,

    fn setup(allocator: std.mem.Allocator, size: Size) !OverheadBaselineCtx {
        const count = size.elementCount();
        var tree = Tree(i64).init(allocator);

        for (0..count) |i| {
            try tree.insert(@intCast(i));
        }

        const queries = try allocator.alloc(i64, count);
        for (0..count) |i| {
            queries[i] = @intCast(i);
        }
        var prng = Prng.init(66666); // Same seed for comparison
        prng.shuffle(i64, queries);

        return .{
            .tree = tree,
            .queries = queries,
            .allocator = allocator,
        };
    }

    fn run(ctx: *OverheadBaselineCtx) void {
        for (ctx.queries) |q| {
            _ = ctx.tree.contains(q);
        }
    }

    fn teardown(ctx: *OverheadBaselineCtx) void {
        ctx.tree.deinit();
        ctx.allocator.free(ctx.queries);
    }
};

// ============================================================================
// CLI Argument Parsing
// ============================================================================

fn parseArgs(allocator: std.mem.Allocator) !Config {
    var config = Config{};

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.skip(); // Skip program name

    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--filter=")) {
            config.filter = arg["--filter=".len..];
        } else if (std.mem.startsWith(u8, arg, "--iterations=")) {
            config.iterations = std.fmt.parseInt(usize, arg["--iterations=".len..], 10) catch 100;
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            config.warmup = std.fmt.parseInt(usize, arg["--warmup=".len..], 10) catch 5;
        } else if (std.mem.startsWith(u8, arg, "--size=")) {
            const sizes_str = arg["--size=".len..];
            config.sizes = .{ .small = false, .medium = false, .large = false };
            var it = std.mem.splitScalar(u8, sizes_str, ',');
            while (it.next()) |s| {
                if (std.mem.eql(u8, s, "S")) config.sizes.small = true;
                if (std.mem.eql(u8, s, "M")) config.sizes.medium = true;
                if (std.mem.eql(u8, s, "L")) config.sizes.large = true;
            }
        } else if (std.mem.eql(u8, arg, "--format=json")) {
            config.format = .json;
        } else if (std.mem.eql(u8, arg, "--format=text")) {
            config.format = .text;
        } else if (std.mem.startsWith(u8, arg, "--threads=")) {
            config.thread_count = std.fmt.parseInt(usize, arg["--threads=".len..], 10) catch 0;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        }
    }

    return config;
}

fn printHelp() void {
    const help =
        \\DIETZ Benchmarks
        \\
        \\Usage: bench [options]
        \\
        \\Options:
        \\  --filter=<pattern>    Run only benchmarks containing pattern
        \\  --iterations=<N>      Number of timed iterations (default: 100)
        \\  --warmup=<N>          Number of warmup iterations (default: 5)
        \\  --size=<S,M,L>        Comma-separated sizes to test (default: S,M,L)
        \\  --format=<text|json>  Output format (default: text)
        \\  --threads=<N>         Thread count for concurrency tests (default: auto)
        \\  --help, -h            Show this help message
        \\
        \\Examples:
        \\  bench                          Run all benchmarks
        \\  bench --filter=insert          Run only insert benchmarks
        \\  bench --size=M --iterations=50 Run medium size with 50 iterations
        \\  bench --format=json            Output JSON for tooling
        \\
    ;
    const stdout = std.fs.File.stdout();
    stdout.writeAll(help) catch {};
}

fn matchesFilter(name: []const u8, filter: ?[]const u8) bool {
    if (filter) |f| {
        return std.mem.indexOf(u8, name, f) != null;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================

fn print(buf: []u8, comptime fmt: []const u8, args: anytype) void {
    const line = std.fmt.bufPrint(buf, fmt, args) catch return;
    std.fs.File.stdout().writeAll(line) catch {};
}

fn write(str: []const u8) void {
    std.fs.File.stdout().writeAll(str) catch {};
}

pub fn main() !void {
    var gpa_state: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const config = try parseArgs(gpa);

    var results: std.ArrayList(BenchmarkResult) = .empty;
    defer results.deinit(gpa);

    var buf: [4096]u8 = undefined;

    if (config.format == .text) {
        write("DIETZ Benchmarks\n");
        write("================\n");
        print(&buf, "Config: iterations={d}, warmup={d}\n\n", .{ config.iterations, config.warmup });
    }

    // Run benchmarks for each enabled size
    const sizes = [_]Size{ .small, .medium, .large };
    const size_enabled = [_]bool{ config.sizes.small, config.sizes.medium, config.sizes.large };

    for (sizes, size_enabled) |size, enabled| {
        if (!enabled) continue;

        if (config.format == .text) {
            print(&buf, "Size: {s} (N={d})\n", .{ size.name(), size.elementCount() });
            write("------------------------------------------------------------\n");
        }

        // Core Operations
        if (matchesFilter("insert/sequential", config.filter)) {
            const r = try runBenchmark(InsertSequentialCtx, "insert/sequential", size, config, gpa, InsertSequentialCtx.setup, InsertSequentialCtx.run, InsertSequentialCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("insert/random", config.filter)) {
            const r = try runBenchmark(InsertRandomCtx, "insert/random", size, config, gpa, InsertRandomCtx.setup, InsertRandomCtx.run, InsertRandomCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("insert/reverse", config.filter)) {
            const r = try runBenchmark(InsertReverseCtx, "insert/reverse", size, config, gpa, InsertReverseCtx.setup, InsertReverseCtx.run, InsertReverseCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("insert/sparse", config.filter)) {
            const r = try runBenchmark(InsertSparseCtx, "insert/sparse", size, config, gpa, InsertSparseCtx.setup, InsertSparseCtx.run, InsertSparseCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("contains/hit", config.filter)) {
            const r = try runBenchmark(ContainsHitCtx, "contains/hit", size, config, gpa, ContainsHitCtx.setup, ContainsHitCtx.run, ContainsHitCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("contains/miss", config.filter)) {
            const r = try runBenchmark(ContainsMissCtx, "contains/miss", size, config, gpa, ContainsMissCtx.setup, ContainsMissCtx.run, ContainsMissCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("delete/sequential", config.filter)) {
            const r = try runBenchmark(DeleteSequentialCtx, "delete/sequential", size, config, gpa, DeleteSequentialCtx.setup, DeleteSequentialCtx.run, DeleteSequentialCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("delete/random", config.filter)) {
            const r = try runBenchmark(DeleteRandomCtx, "delete/random", size, config, gpa, DeleteRandomCtx.setup, DeleteRandomCtx.run, DeleteRandomCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("delete/split", config.filter)) {
            const r = try runBenchmark(DeleteSplitCtx, "delete/split", size, config, gpa, DeleteSplitCtx.setup, DeleteSplitCtx.run, DeleteSplitCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        // Range Operations
        if (matchesFilter("insertRange/small", config.filter)) {
            const r = try runBenchmark(InsertRangeSmallCtx, "insertRange/small", size, config, gpa, InsertRangeSmallCtx.setup, InsertRangeSmallCtx.run, InsertRangeSmallCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("insertRange/large", config.filter)) {
            const r = try runBenchmark(InsertRangeLargeCtx, "insertRange/large", size, config, gpa, InsertRangeLargeCtx.setup, InsertRangeLargeCtx.run, InsertRangeLargeCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("containsRange/hit", config.filter)) {
            const r = try runBenchmark(ContainsRangeHitCtx, "containsRange/hit", size, config, gpa, ContainsRangeHitCtx.setup, ContainsRangeHitCtx.run, ContainsRangeHitCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("containsRange/miss", config.filter)) {
            const r = try runBenchmark(ContainsRangeMissCtx, "containsRange/miss", size, config, gpa, ContainsRangeMissCtx.setup, ContainsRangeMissCtx.run, ContainsRangeMissCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("intersects", config.filter)) {
            const r = try runBenchmark(IntersectsCtx, "intersects", size, config, gpa, IntersectsCtx.setup, IntersectsCtx.run, IntersectsCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("deleteRange", config.filter)) {
            const r = try runBenchmark(DeleteRangeCtx, "deleteRange", size, config, gpa, DeleteRangeCtx.setup, DeleteRangeCtx.run, DeleteRangeCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        // Set Operations (skip for large size - too slow)
        if (size != .large) {
            if (matchesFilter("setUnion/disjoint", config.filter)) {
                const r = try runBenchmark(SetUnionDisjointCtx, "setUnion/disjoint", size, config, gpa, SetUnionDisjointCtx.setup, SetUnionDisjointCtx.run, SetUnionDisjointCtx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }

            if (matchesFilter("setUnion/overlapping", config.filter)) {
                const r = try runBenchmark(SetUnionOverlappingCtx, "setUnion/overlapping", size, config, gpa, SetUnionOverlappingCtx.setup, SetUnionOverlappingCtx.run, SetUnionOverlappingCtx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }

            if (matchesFilter("setIntersection", config.filter)) {
                const r = try runBenchmark(SetIntersectionCtx, "setIntersection", size, config, gpa, SetIntersectionCtx.setup, SetIntersectionCtx.run, SetIntersectionCtx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }

            if (matchesFilter("setDifference", config.filter)) {
                const r = try runBenchmark(SetDifferenceCtx, "setDifference", size, config, gpa, SetDifferenceCtx.setup, SetDifferenceCtx.run, SetDifferenceCtx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }
        }

        // Traversal Operations
        if (matchesFilter("iterator", config.filter)) {
            const r = try runBenchmark(IteratorCtx, "iterator", size, config, gpa, IteratorCtx.setup, IteratorCtx.run, IteratorCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("intervals", config.filter)) {
            const r = try runBenchmark(IntervalsCtx, "intervals", size, config, gpa, IntervalsCtx.setup, IntervalsCtx.run, IntervalsCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        if (matchesFilter("count", config.filter)) {
            const r = try runBenchmark(CountCtx, "count", size, config, gpa, CountCtx.setup, CountCtx.run, CountCtx.teardown);
            try results.append(gpa, r);
            if (config.format == .text) printTextResult(r, &buf);
        }

        // Concurrency Operations (skip for large size)
        if (size != .large) {
            if (matchesFilter("concurrent/read_only", config.filter)) {
                const r = try runBenchmark(ConcurrentReadCtx, "concurrent/read_only", size, config, gpa, ConcurrentReadCtx.setup, ConcurrentReadCtx.run, ConcurrentReadCtx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }

            if (matchesFilter("concurrent/write_only", config.filter)) {
                const r = try runBenchmark(ConcurrentWriteCtx, "concurrent/write_only", size, config, gpa, ConcurrentWriteCtx.setup, ConcurrentWriteCtx.run, ConcurrentWriteCtx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }

            if (matchesFilter("concurrent/mixed_90_10", config.filter)) {
                const r = try runBenchmark(ConcurrentMixed90_10Ctx, "concurrent/mixed_90_10", size, config, gpa, ConcurrentMixed90_10Ctx.setup, ConcurrentMixed90_10Ctx.run, ConcurrentMixed90_10Ctx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }

            if (matchesFilter("overhead/baseline", config.filter)) {
                const r = try runBenchmark(OverheadBaselineCtx, "overhead/baseline", size, config, gpa, OverheadBaselineCtx.setup, OverheadBaselineCtx.run, OverheadBaselineCtx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }

            if (matchesFilter("overhead/threadsafe", config.filter)) {
                const r = try runBenchmark(OverheadCtx, "overhead/threadsafe", size, config, gpa, OverheadCtx.setup, OverheadCtx.run, OverheadCtx.teardown);
                try results.append(gpa, r);
                if (config.format == .text) printTextResult(r, &buf);
            }
        }

        if (config.format == .text) {
            write("\n");
        }
    }

    // JSON output
    if (config.format == .json) {
        printJsonResults(results.items, config, &buf);
    }
}
