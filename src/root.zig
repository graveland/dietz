//! Discrete Interval Encoding Tree (DIET)
//!
//! A data structure for efficiently storing sets of integers using maximal
//! contiguous intervals in a binary search tree. Based on "Diets for Fat Sets"
//! by Martin Erwig.
//!
//! Provides both thread-safe (`ThreadSafeTree`) and non-thread-safe (`Tree`) variants.
//!
//! ## Example
//! ```zig
//! var tree = Tree(i32).init(allocator);
//! defer tree.deinit();
//!
//! try tree.insert(5);
//! try tree.insert(6);
//! try tree.insert(7);
//!
//! // These are now merged into a single interval [5, 7]
//! std.debug.assert(tree.contains(6));
//! std.debug.assert(tree.intervalCount() == 1);
//! ```

const std = @import("std");

fn isInteger(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => true,
        else => false,
    };
}

/// Represents a contiguous range of consecutive integers [first, last]
pub fn Interval(comptime T: type) type {
    comptime {
        if (!isInteger(T)) {
            @compileError("Interval requires an integer type");
        }
    }

    return struct {
        first: T,
        last: T,

        const Self = @This();

        /// Create a single-element interval
        pub fn init(elem: T) Self {
            return .{ .first = elem, .last = elem };
        }

        /// Create an interval from a range [first, last]
        /// Asserts that first <= last in debug builds
        pub fn initRange(first: T, last: T) Self {
            std.debug.assert(first <= last);
            return .{ .first = first, .last = last };
        }

        /// Check if element is within this interval
        pub fn has(self: Self, elem: T) bool {
            return elem >= self.first and elem <= self.last;
        }

        /// Check if the entire range [first, last] is within this interval
        pub fn hasRange(self: Self, first: T, last: T) bool {
            return first >= self.first and last <= self.last;
        }

        /// Check if this interval overlaps with the range [first, last]
        pub fn intersectsRange(self: Self, first: T, last: T) bool {
            return self.first <= last and self.last >= first;
        }

        /// Check if element is adjacent to the left of this interval (elem + 1 == first)
        pub fn adjacentLeft(self: Self, elem: T) bool {
            if (self.first == std.math.minInt(T)) return false;
            return elem == self.first - 1;
        }

        /// Check if element is adjacent to the right of this interval (elem - 1 == last)
        pub fn adjacentRight(self: Self, elem: T) bool {
            if (self.last == std.math.maxInt(T)) return false;
            return elem == self.last + 1;
        }

        /// Check if this interval is adjacent to another interval
        pub fn adjacent(self: Self, other: Self) bool {
            // Check if other.last + 1 == self.first (other is to the left)
            if (other.last != std.math.maxInt(T) and other.last + 1 == self.first) {
                return true;
            }
            // Check if self.last + 1 == other.first (other is to the right)
            if (self.last != std.math.maxInt(T) and self.last + 1 == other.first) {
                return true;
            }
            return false;
        }

        /// Merge another interval into this one
        /// Asserts that the intervals are adjacent or overlapping in debug builds
        pub fn merge(self: *Self, other: Self) void {
            std.debug.assert(self.adjacent(other) or self.intersectsRange(other.first, other.last));
            self.first = @min(self.first, other.first);
            self.last = @max(self.last, other.last);
        }

        /// Returns the number of elements in this interval
        pub fn count(self: Self) usize {
            return @as(usize, @intCast(self.last - self.first)) + 1;
        }
    };
}

/// Non-thread-safe Discrete Interval Encoding Tree
pub fn Tree(comptime T: type) type {
    comptime {
        if (!isInteger(T)) {
            @compileError("Tree requires an integer type");
        }
    }

    return struct {
        const Self = @This();
        pub const IntervalType = Interval(T);

        pub const Node = struct {
            interval: IntervalType,
            left: ?*Node = null,
            right: ?*Node = null,
        };

        allocator: std.mem.Allocator,
        root: ?*Node = null,
        node_count: usize = 0,

        /// Initialize an empty tree
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Free all nodes in the tree (iterative to avoid stack overflow)
        pub fn deinit(self: *Self) void {
            // Use iterative post-order traversal with explicit stack
            var stack: std.ArrayList(*Node) = .empty;
            defer stack.deinit(self.allocator);

            var current = self.root;
            var last_visited: ?*Node = null;

            while (current != null or stack.items.len > 0) {
                if (current) |node| {
                    stack.append(self.allocator, node) catch {
                        // If we can't allocate stack space, fall back to recursive
                        self.freeNodeRecursive(self.root);
                        self.root = null;
                        self.node_count = 0;
                        return;
                    };
                    current = node.left;
                } else {
                    const peek_node = stack.items[stack.items.len - 1];
                    if (peek_node.right != null and last_visited != peek_node.right) {
                        current = peek_node.right;
                    } else {
                        _ = stack.pop();
                        last_visited = peek_node;
                        self.allocator.destroy(peek_node);
                    }
                }
            }

            self.root = null;
            self.node_count = 0;
        }

        fn freeNodeRecursive(self: *Self, node: ?*Node) void {
            if (node) |n| {
                self.freeNodeRecursive(n.left);
                self.freeNodeRecursive(n.right);
                self.allocator.destroy(n);
            }
        }

        /// Returns true if the tree contains no elements
        pub fn isEmpty(self: *const Self) bool {
            return self.root == null;
        }

        /// Returns the number of intervals (nodes) in the tree
        pub fn intervalCount(self: *const Self) usize {
            return self.node_count;
        }

        /// Returns the total number of elements across all intervals
        pub fn count(self: *const Self) usize {
            var total: usize = 0;
            var it = self.iterator();
            while (it.next()) |interval| {
                total += interval.count();
            }
            return total;
        }

        /// Check if an element exists in the tree (iterative)
        pub fn contains(self: *const Self, elem: T) bool {
            var current = self.root;
            while (current) |node| {
                if (node.interval.has(elem)) {
                    return true;
                } else if (elem < node.interval.first) {
                    current = node.left;
                } else {
                    current = node.right;
                }
            }
            return false;
        }

        /// Check if the entire range [first, last] exists in the tree
        pub fn containsRange(self: *const Self, first: T, last: T) bool {
            std.debug.assert(first <= last);
            var current = self.root;
            while (current) |node| {
                if (node.interval.hasRange(first, last)) {
                    return true;
                } else if (last < node.interval.first) {
                    current = node.left;
                } else if (first > node.interval.last) {
                    current = node.right;
                } else {
                    // Range partially overlaps this interval but isn't fully contained
                    return false;
                }
            }
            return false;
        }

        /// Check if any element in the range [first, last] exists in the tree
        pub fn intersects(self: *const Self, first: T, last: T) bool {
            std.debug.assert(first <= last);
            var current = self.root;
            while (current) |node| {
                if (node.interval.intersectsRange(first, last)) {
                    return true;
                } else if (last < node.interval.first) {
                    current = node.left;
                } else {
                    current = node.right;
                }
            }
            return false;
        }

        /// Insert an element into the tree
        pub fn insert(self: *Self, elem: T) !void {
            self.root = try self.insertInto(self.root, elem);
        }

        /// Insert a range of elements [first, last] into the tree
        pub fn insertRange(self: *Self, first: T, last: T) !void {
            std.debug.assert(first <= last);
            // Insert each element - the merging will combine them
            // This is not optimal but correct; a more efficient implementation
            // would insert the range as a single interval and merge
            var elem = first;
            while (elem <= last) : (elem += 1) {
                try self.insert(elem);
                if (elem == last) break; // Prevent overflow
            }
        }

        fn insertInto(self: *Self, node: ?*Node, elem: T) !?*Node {
            const n = node orelse {
                const new_node = try self.allocator.create(Node);
                new_node.* = .{ .interval = IntervalType.init(elem) };
                self.node_count += 1;
                return new_node;
            };

            if (n.interval.has(elem)) {
                return n;
            } else if (n.interval.adjacentLeft(elem)) {
                n.interval.first = elem;
                self.joinLeft(n);
                return n;
            } else if (n.interval.adjacentRight(elem)) {
                n.interval.last = elem;
                self.joinRight(n);
                return n;
            } else if (elem < n.interval.first) {
                n.left = try self.insertInto(n.left, elem);
                return n;
            } else {
                n.right = try self.insertInto(n.right, elem);
                return n;
            }
        }

        /// Delete an element from the tree
        pub fn delete(self: *Self, elem: T) void {
            self.root = self.deleteFrom(self.root, elem);
        }

        /// Delete a range of elements [first, last] from the tree
        pub fn deleteRange(self: *Self, first: T, last: T) void {
            std.debug.assert(first <= last);
            var elem = first;
            while (elem <= last) : (elem += 1) {
                self.delete(elem);
                if (elem == last) break;
            }
        }

        fn deleteFrom(self: *Self, node: ?*Node, elem: T) ?*Node {
            const n = node orelse return null;

            if (elem < n.interval.first) {
                n.left = self.deleteFrom(n.left, elem);
                return n;
            } else if (elem > n.interval.last) {
                n.right = self.deleteFrom(n.right, elem);
                return n;
            } else {
                // Element is within this interval
                if (n.interval.first == n.interval.last) {
                    // Single element interval - remove the node
                    const result = self.mergeChildren(n.left, n.right);
                    self.allocator.destroy(n);
                    self.node_count -= 1;
                    return result;
                } else if (elem == n.interval.first) {
                    // Remove from left edge
                    n.interval.first += 1;
                    return n;
                } else if (elem == n.interval.last) {
                    // Remove from right edge
                    n.interval.last -= 1;
                    return n;
                } else {
                    // Split the interval: [first, elem-1] and [elem+1, last]
                    // Current node keeps [first, elem-1]
                    // Create new node for [elem+1, last] and insert into right subtree
                    const new_interval = IntervalType.initRange(elem + 1, n.interval.last);
                    n.interval.last = elem - 1;

                    const new_node = self.allocator.create(Node) catch {
                        // If allocation fails, we've already modified the interval
                        // This leaves the tree in an inconsistent state - restore
                        n.interval.last = elem - 1; // Already set
                        return n; // Best effort
                    };
                    new_node.* = .{ .interval = new_interval, .left = null, .right = n.right };
                    n.right = new_node;
                    self.node_count += 1;
                    return n;
                }
            }
        }

        fn mergeChildren(self: *Self, left: ?*Node, right: ?*Node) ?*Node {
            _ = self;
            if (left == null) return right;
            if (right == null) return left;

            // Find rightmost node in left subtree
            var current = left.?;
            while (current.right) |r| {
                current = r;
            }
            current.right = right;
            return left;
        }

        fn joinLeft(self: *Self, node: *Node) void {
            if (node.left == null) return;

            var parent: ?*Node = null;
            var current = node.left.?;
            while (current.right) |right| {
                parent = current;
                current = right;
            }

            if (current.interval.last != std.math.maxInt(T) and
                current.interval.last + 1 == node.interval.first)
            {
                node.interval.first = current.interval.first;

                if (parent) |p| {
                    p.right = current.left;
                } else {
                    node.left = current.left;
                }
                self.allocator.destroy(current);
                self.node_count -= 1;
            }
        }

        fn joinRight(self: *Self, node: *Node) void {
            if (node.right == null) return;

            var parent: ?*Node = null;
            var current = node.right.?;
            while (current.left) |left| {
                parent = current;
                current = left;
            }

            if (node.interval.last != std.math.maxInt(T) and
                node.interval.last + 1 == current.interval.first)
            {
                node.interval.last = current.interval.last;

                if (parent) |p| {
                    p.left = current.right;
                } else {
                    node.right = current.right;
                }
                self.allocator.destroy(current);
                self.node_count -= 1;
            }
        }

        /// Get all intervals in the tree (in order)
        /// Caller owns the returned slice and must free it with the same allocator
        pub fn intervals(self: *const Self, allocator: std.mem.Allocator) ![]IntervalType {
            var list: std.ArrayList(IntervalType) = .empty;
            errdefer list.deinit(allocator);

            var it = self.iterator();
            while (it.next()) |interval| {
                try list.append(allocator, interval);
            }
            return list.toOwnedSlice(allocator);
        }

        /// Iterator for traversing intervals without allocation
        pub const Iterator = struct {
            stack: [64]StackEntry = undefined,
            stack_len: usize = 0,

            const StackEntry = struct {
                node: *Node,
                visited_left: bool,
            };

            pub fn next(self: *Iterator) ?IntervalType {
                while (self.stack_len > 0) {
                    const top = &self.stack[self.stack_len - 1];

                    if (!top.visited_left) {
                        top.visited_left = true;
                        if (top.node.left) |left| {
                            if (self.stack_len < self.stack.len) {
                                self.stack[self.stack_len] = .{ .node = left, .visited_left = false };
                                self.stack_len += 1;
                            }
                        }
                        continue;
                    }

                    const interval = top.node.interval;
                    const right = top.node.right;
                    self.stack_len -= 1;

                    if (right) |r| {
                        if (self.stack_len < self.stack.len) {
                            self.stack[self.stack_len] = .{ .node = r, .visited_left = false };
                            self.stack_len += 1;
                        }
                    }

                    return interval;
                }
                return null;
            }
        };

        /// Returns an iterator over all intervals in order
        /// This is allocation-free but limited to trees of depth <= 64
        pub fn iterator(self: *const Self) Iterator {
            var it = Iterator{};
            if (self.root) |root| {
                it.stack[0] = .{ .node = root, .visited_left = false };
                it.stack_len = 1;
            }
            return it;
        }

        /// Create a new tree that is the union of this tree and another
        /// Caller owns the returned tree
        pub fn setUnion(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !Self {
            var result = Self.init(allocator);
            errdefer result.deinit();

            var it = self.iterator();
            while (it.next()) |interval| {
                var elem = interval.first;
                while (elem <= interval.last) : (elem += 1) {
                    try result.insert(elem);
                    if (elem == interval.last) break;
                }
            }

            var other_it = other.iterator();
            while (other_it.next()) |interval| {
                var elem = interval.first;
                while (elem <= interval.last) : (elem += 1) {
                    try result.insert(elem);
                    if (elem == interval.last) break;
                }
            }

            return result;
        }

        /// Create a new tree that is the intersection of this tree and another
        /// Caller owns the returned tree
        pub fn setIntersection(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !Self {
            var result = Self.init(allocator);
            errdefer result.deinit();

            var it = self.iterator();
            while (it.next()) |interval| {
                var elem = interval.first;
                while (elem <= interval.last) : (elem += 1) {
                    if (other.contains(elem)) {
                        try result.insert(elem);
                    }
                    if (elem == interval.last) break;
                }
            }

            return result;
        }

        /// Create a new tree that is this tree minus the other tree (difference)
        /// Caller owns the returned tree
        pub fn setDifference(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !Self {
            var result = Self.init(allocator);
            errdefer result.deinit();

            var it = self.iterator();
            while (it.next()) |interval| {
                var elem = interval.first;
                while (elem <= interval.last) : (elem += 1) {
                    if (!other.contains(elem)) {
                        try result.insert(elem);
                    }
                    if (elem == interval.last) break;
                }
            }

            return result;
        }
    };
}

/// Thread-safe Discrete Interval Encoding Tree
/// Wraps the non-thread-safe Tree with a read-write lock for safe concurrent access.
/// Multiple readers can access the tree concurrently, but writers get exclusive access.
pub fn ThreadSafeTree(comptime T: type) type {
    comptime {
        if (!isInteger(T)) {
            @compileError("ThreadSafeTree requires an integer type");
        }
    }

    return struct {
        const Self = @This();
        const InnerTree = Tree(T);
        pub const IntervalType = InnerTree.IntervalType;
        pub const Node = InnerTree.Node;

        inner: InnerTree,
        rwlock: std.Thread.RwLock = .{},

        /// Initialize an empty thread-safe tree
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .inner = InnerTree.init(allocator) };
        }

        /// Free all nodes in the tree (acquires write lock)
        pub fn deinit(self: *Self) void {
            self.rwlock.lock();
            defer self.rwlock.unlock();
            self.inner.deinit();
        }

        /// Returns true if the tree contains no elements
        pub fn isEmpty(self: *Self) bool {
            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();
            return self.inner.isEmpty();
        }

        /// Returns the number of intervals (nodes) in the tree
        pub fn intervalCount(self: *Self) usize {
            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();
            return self.inner.intervalCount();
        }

        /// Returns the total number of elements across all intervals
        pub fn count(self: *Self) usize {
            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();
            return self.inner.count();
        }

        /// Check if an element exists in the tree (acquires read lock)
        pub fn contains(self: *Self, elem: T) bool {
            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();
            return self.inner.contains(elem);
        }

        /// Check if the entire range [first, last] exists in the tree
        pub fn containsRange(self: *Self, first: T, last: T) bool {
            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();
            return self.inner.containsRange(first, last);
        }

        /// Check if any element in the range [first, last] exists in the tree
        pub fn intersects(self: *Self, first: T, last: T) bool {
            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();
            return self.inner.intersects(first, last);
        }

        /// Insert an element into the tree (acquires write lock)
        pub fn insert(self: *Self, elem: T) !void {
            self.rwlock.lock();
            defer self.rwlock.unlock();
            try self.inner.insert(elem);
        }

        /// Insert a range of elements [first, last] into the tree
        pub fn insertRange(self: *Self, first: T, last: T) !void {
            self.rwlock.lock();
            defer self.rwlock.unlock();
            try self.inner.insertRange(first, last);
        }

        /// Delete an element from the tree (acquires write lock)
        pub fn delete(self: *Self, elem: T) void {
            self.rwlock.lock();
            defer self.rwlock.unlock();
            self.inner.delete(elem);
        }

        /// Delete a range of elements [first, last] from the tree
        pub fn deleteRange(self: *Self, first: T, last: T) void {
            self.rwlock.lock();
            defer self.rwlock.unlock();
            self.inner.deleteRange(first, last);
        }

        /// Get all intervals in the tree (acquires read lock)
        pub fn intervals(self: *Self, allocator: std.mem.Allocator) ![]IntervalType {
            self.rwlock.lockShared();
            defer self.rwlock.unlockShared();
            return self.inner.intervals(allocator);
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "empty tree contains nothing" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try std.testing.expect(!tree.contains(0));
    try std.testing.expect(!tree.contains(42));
    try std.testing.expect(!tree.contains(-100));
    try std.testing.expect(tree.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), tree.intervalCount());
    try std.testing.expectEqual(@as(usize, 0), tree.count());
}

test "single element" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);

    try std.testing.expect(tree.contains(5));
    try std.testing.expect(!tree.contains(4));
    try std.testing.expect(!tree.contains(6));
    try std.testing.expect(!tree.isEmpty());
    try std.testing.expectEqual(@as(usize, 1), tree.intervalCount());
    try std.testing.expectEqual(@as(usize, 1), tree.count());
}

test "membership - left edge" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);

    try std.testing.expect(tree.contains(5));
    try std.testing.expect(!tree.contains(4));
}

test "membership - right edge" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);

    try std.testing.expect(tree.contains(7));
    try std.testing.expect(!tree.contains(8));
}

test "membership - internal" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);

    try std.testing.expect(tree.contains(6));
}

test "left child traversal" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(10);
    try tree.insert(5);

    try std.testing.expect(tree.contains(10));
    try std.testing.expect(tree.contains(5));
    try std.testing.expect(!tree.contains(7));
}

test "right child traversal" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(10);
    try tree.insert(15);

    try std.testing.expect(tree.contains(10));
    try std.testing.expect(tree.contains(15));
    try std.testing.expect(!tree.contains(12));
}

test "simple merge - adjacent elements" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i32, 5), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 6), ivs[0].last);
    try std.testing.expectEqual(@as(usize, 1), tree.intervalCount());
    try std.testing.expectEqual(@as(usize, 2), tree.count());
}

test "simple merge - reverse order" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(6);
    try tree.insert(5);

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i32, 5), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 6), ivs[0].last);
}

test "LL merge" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(10);
    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);
    try tree.insert(8);
    try tree.insert(9);

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i32, 5), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 10), ivs[0].last);
}

test "RR merge" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(10);
    try tree.insert(15);
    try tree.insert(14);
    try tree.insert(13);
    try tree.insert(12);
    try tree.insert(11);

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i32, 10), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 15), ivs[0].last);
}

test "LR merge" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(10);
    try tree.insert(5);
    try tree.insert(7);
    try tree.insert(6);

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 2), ivs.len);
    try std.testing.expectEqual(@as(i32, 5), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 7), ivs[0].last);
}

test "RL merge" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(10);
    try tree.insert(15);
    try tree.insert(13);
    try tree.insert(14);

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 2), ivs.len);
    try std.testing.expectEqual(@as(i32, 13), ivs[1].first);
    try std.testing.expectEqual(@as(i32, 15), ivs[1].last);
}

test "duplicate insert" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(5);
    try tree.insert(5);

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i32, 5), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 5), ivs[0].last);
}

test "unsigned integers" {
    const allocator = std.testing.allocator;
    var tree = Tree(u64).init(allocator);
    defer tree.deinit();

    try tree.insert(100);
    try tree.insert(101);
    try tree.insert(102);

    try std.testing.expect(tree.contains(100));
    try std.testing.expect(tree.contains(101));
    try std.testing.expect(tree.contains(102));
    try std.testing.expect(!tree.contains(99));
    try std.testing.expect(!tree.contains(103));
}

test "thread-safe tree basic operations" {
    const allocator = std.testing.allocator;
    var tree = ThreadSafeTree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);

    try std.testing.expect(tree.contains(5));
    try std.testing.expect(tree.contains(6));
    try std.testing.expect(tree.contains(7));
    try std.testing.expect(!tree.contains(4));
    try std.testing.expect(!tree.contains(8));
}

test "interval adjacency" {
    const I = Interval(i32);

    const a = I.initRange(1, 5);
    const b = I.initRange(6, 10);
    const c = I.initRange(8, 12);

    try std.testing.expect(a.adjacent(b));
    try std.testing.expect(b.adjacent(a));
    try std.testing.expect(!a.adjacent(c));
    try std.testing.expect(!b.adjacent(c));
}

test "interval has" {
    const I = Interval(i32);
    const iv = I.initRange(5, 10);

    try std.testing.expect(!iv.has(4));
    try std.testing.expect(iv.has(5));
    try std.testing.expect(iv.has(7));
    try std.testing.expect(iv.has(10));
    try std.testing.expect(!iv.has(11));
}

test "delete single element" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try std.testing.expect(tree.contains(5));

    tree.delete(5);
    try std.testing.expect(!tree.contains(5));
    try std.testing.expect(tree.isEmpty());
}

test "delete from left edge of interval" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);

    tree.delete(5);

    try std.testing.expect(!tree.contains(5));
    try std.testing.expect(tree.contains(6));
    try std.testing.expect(tree.contains(7));

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i32, 6), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 7), ivs[0].last);
}

test "delete from right edge of interval" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);

    tree.delete(7);

    try std.testing.expect(tree.contains(5));
    try std.testing.expect(tree.contains(6));
    try std.testing.expect(!tree.contains(7));

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i32, 5), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 6), ivs[0].last);
}

test "delete from middle of interval - splits" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);
    try tree.insert(8);
    try tree.insert(9);

    tree.delete(7);

    try std.testing.expect(tree.contains(5));
    try std.testing.expect(tree.contains(6));
    try std.testing.expect(!tree.contains(7));
    try std.testing.expect(tree.contains(8));
    try std.testing.expect(tree.contains(9));

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 2), ivs.len);
    try std.testing.expectEqual(@as(i32, 5), ivs[0].first);
    try std.testing.expectEqual(@as(i32, 6), ivs[0].last);
    try std.testing.expectEqual(@as(i32, 8), ivs[1].first);
    try std.testing.expectEqual(@as(i32, 9), ivs[1].last);
}

test "delete non-existent element" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);

    tree.delete(100);

    try std.testing.expect(tree.contains(5));
}

test "containsRange" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);
    try tree.insert(8);
    try tree.insert(9);

    try std.testing.expect(tree.containsRange(5, 9));
    try std.testing.expect(tree.containsRange(6, 8));
    try std.testing.expect(tree.containsRange(5, 5));
    try std.testing.expect(!tree.containsRange(4, 9));
    try std.testing.expect(!tree.containsRange(5, 10));
    try std.testing.expect(!tree.containsRange(1, 3));
}

test "intersects" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);

    try std.testing.expect(tree.intersects(5, 7));
    try std.testing.expect(tree.intersects(4, 6));
    try std.testing.expect(tree.intersects(6, 10));
    try std.testing.expect(tree.intersects(1, 100));
    try std.testing.expect(!tree.intersects(1, 4));
    try std.testing.expect(!tree.intersects(8, 10));
}

test "insertRange" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insertRange(5, 10);

    try std.testing.expect(tree.contains(5));
    try std.testing.expect(tree.contains(7));
    try std.testing.expect(tree.contains(10));
    try std.testing.expect(!tree.contains(4));
    try std.testing.expect(!tree.contains(11));
    try std.testing.expectEqual(@as(usize, 6), tree.count());
}

test "deleteRange" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insertRange(1, 10);
    tree.deleteRange(4, 7);

    try std.testing.expect(tree.contains(1));
    try std.testing.expect(tree.contains(3));
    try std.testing.expect(!tree.contains(4));
    try std.testing.expect(!tree.contains(7));
    try std.testing.expect(tree.contains(8));
    try std.testing.expect(tree.contains(10));
}

test "iterator" {
    const allocator = std.testing.allocator;
    var tree = Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(1);
    try tree.insert(2);
    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(10);

    var it = tree.iterator();
    const first = it.next().?;
    try std.testing.expectEqual(@as(i32, 1), first.first);
    try std.testing.expectEqual(@as(i32, 2), first.last);

    const second = it.next().?;
    try std.testing.expectEqual(@as(i32, 5), second.first);
    try std.testing.expectEqual(@as(i32, 6), second.last);

    const third = it.next().?;
    try std.testing.expectEqual(@as(i32, 10), third.first);
    try std.testing.expectEqual(@as(i32, 10), third.last);

    try std.testing.expect(it.next() == null);
}

test "setUnion" {
    const allocator = std.testing.allocator;
    var tree1 = Tree(i32).init(allocator);
    defer tree1.deinit();
    var tree2 = Tree(i32).init(allocator);
    defer tree2.deinit();

    try tree1.insert(1);
    try tree1.insert(2);
    try tree1.insert(5);

    try tree2.insert(2);
    try tree2.insert(3);
    try tree2.insert(6);

    var result = try tree1.setUnion(&tree2, allocator);
    defer result.deinit();

    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(!result.contains(4));
    try std.testing.expect(result.contains(5));
    try std.testing.expect(result.contains(6));
}

test "setIntersection" {
    const allocator = std.testing.allocator;
    var tree1 = Tree(i32).init(allocator);
    defer tree1.deinit();
    var tree2 = Tree(i32).init(allocator);
    defer tree2.deinit();

    try tree1.insertRange(1, 5);
    try tree2.insertRange(3, 7);

    var result = try tree1.setIntersection(&tree2, allocator);
    defer result.deinit();

    try std.testing.expect(!result.contains(1));
    try std.testing.expect(!result.contains(2));
    try std.testing.expect(result.contains(3));
    try std.testing.expect(result.contains(4));
    try std.testing.expect(result.contains(5));
    try std.testing.expect(!result.contains(6));
    try std.testing.expect(!result.contains(7));
}

test "setDifference" {
    const allocator = std.testing.allocator;
    var tree1 = Tree(i32).init(allocator);
    defer tree1.deinit();
    var tree2 = Tree(i32).init(allocator);
    defer tree2.deinit();

    try tree1.insertRange(1, 5);
    try tree2.insertRange(3, 7);

    var result = try tree1.setDifference(&tree2, allocator);
    defer result.deinit();

    try std.testing.expect(result.contains(1));
    try std.testing.expect(result.contains(2));
    try std.testing.expect(!result.contains(3));
    try std.testing.expect(!result.contains(4));
    try std.testing.expect(!result.contains(5));
}

test "minInt boundary - unsigned" {
    const allocator = std.testing.allocator;
    var tree = Tree(u8).init(allocator);
    defer tree.deinit();

    try tree.insert(0);
    try tree.insert(1);
    try tree.insert(2);

    try std.testing.expect(tree.contains(0));
    try std.testing.expect(tree.contains(1));
    try std.testing.expect(tree.contains(2));

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(u8, 0), ivs[0].first);
    try std.testing.expectEqual(@as(u8, 2), ivs[0].last);
}

test "maxInt boundary - unsigned" {
    const allocator = std.testing.allocator;
    var tree = Tree(u8).init(allocator);
    defer tree.deinit();

    try tree.insert(253);
    try tree.insert(254);
    try tree.insert(255);

    try std.testing.expect(tree.contains(253));
    try std.testing.expect(tree.contains(254));
    try std.testing.expect(tree.contains(255));

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(u8, 253), ivs[0].first);
    try std.testing.expectEqual(@as(u8, 255), ivs[0].last);
}

test "minInt boundary - signed" {
    const allocator = std.testing.allocator;
    var tree = Tree(i8).init(allocator);
    defer tree.deinit();

    try tree.insert(-128);
    try tree.insert(-127);
    try tree.insert(-126);

    try std.testing.expect(tree.contains(-128));
    try std.testing.expect(tree.contains(-127));
    try std.testing.expect(tree.contains(-126));

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i8, -128), ivs[0].first);
    try std.testing.expectEqual(@as(i8, -126), ivs[0].last);
}

test "maxInt boundary - signed" {
    const allocator = std.testing.allocator;
    var tree = Tree(i8).init(allocator);
    defer tree.deinit();

    try tree.insert(125);
    try tree.insert(126);
    try tree.insert(127);

    try std.testing.expect(tree.contains(125));
    try std.testing.expect(tree.contains(126));
    try std.testing.expect(tree.contains(127));

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    try std.testing.expectEqual(@as(usize, 1), ivs.len);
    try std.testing.expectEqual(@as(i8, 125), ivs[0].first);
    try std.testing.expectEqual(@as(i8, 127), ivs[0].last);
}

test "interval count" {
    const I = Interval(i32);

    const single = I.init(5);
    try std.testing.expectEqual(@as(usize, 1), single.count());

    const range = I.initRange(5, 10);
    try std.testing.expectEqual(@as(usize, 6), range.count());
}

test "thread-safe delete" {
    const allocator = std.testing.allocator;
    var tree = ThreadSafeTree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(5);
    try tree.insert(6);
    try tree.insert(7);

    tree.delete(6);

    try std.testing.expect(tree.contains(5));
    try std.testing.expect(!tree.contains(6));
    try std.testing.expect(tree.contains(7));
}

test "thread-safe utility methods" {
    const allocator = std.testing.allocator;
    var tree = ThreadSafeTree(i32).init(allocator);
    defer tree.deinit();

    try std.testing.expect(tree.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), tree.count());

    try tree.insertRange(1, 5);

    try std.testing.expect(!tree.isEmpty());
    try std.testing.expectEqual(@as(usize, 5), tree.count());
    try std.testing.expect(tree.containsRange(1, 5));
    try std.testing.expect(tree.intersects(3, 10));
}
