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
        /// Uses wider arithmetic to avoid overflow with signed types
        pub fn count(self: Self) usize {
            // Cast to wider signed type to avoid overflow when last - first
            // exceeds the range of T (e.g., i8: 127 - (-128) = 255 overflows i8)
            const first_wide: i128 = self.first;
            const last_wide: i128 = self.last;
            return @intCast(last_wide - first_wide + 1);
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
            height: i32 = 1,
        };

        // AVL helper functions
        fn nodeHeight(node: ?*Node) i32 {
            return if (node) |n| n.height else 0;
        }

        fn updateHeight(node: *Node) void {
            node.height = 1 + @max(nodeHeight(node.left), nodeHeight(node.right));
        }

        fn balanceFactor(node: *Node) i32 {
            return nodeHeight(node.left) - nodeHeight(node.right);
        }

        fn rotateRight(self: *Self, y: *Node) *Node {
            _ = self;
            const x = y.left.?;
            const t2 = x.right;

            x.right = y;
            y.left = t2;

            updateHeight(y);
            updateHeight(x);

            return x;
        }

        fn rotateLeft(self: *Self, x: *Node) *Node {
            _ = self;
            const y = x.right.?;
            const t2 = y.left;

            y.left = x;
            x.right = t2;

            updateHeight(x);
            updateHeight(y);

            return y;
        }

        fn rebalance(self: *Self, node: *Node) *Node {
            updateHeight(node);
            const balance = balanceFactor(node);

            // Left heavy
            if (balance > 1) {
                if (balanceFactor(node.left.?) < 0) {
                    // Left-Right case
                    node.left = self.rotateLeft(node.left.?);
                }
                // Left-Left case
                return self.rotateRight(node);
            }

            // Right heavy
            if (balance < -1) {
                if (balanceFactor(node.right.?) > 0) {
                    // Right-Left case
                    node.right = self.rotateRight(node.right.?);
                }
                // Right-Right case
                return self.rotateLeft(node);
            }

            return node;
        }

        fn debugValidateTree(self: *Self, node: ?*Node) void {
            const n = node orelse return;

            // Validate height is correct
            const left_h = nodeHeight(n.left);
            const right_h = nodeHeight(n.right);
            const expected_h = 1 + @max(left_h, right_h);
            std.debug.assert(n.height == expected_h);

            // Validate AVL property
            const balance = left_h - right_h;
            std.debug.assert(balance >= -1 and balance <= 1);

            // Validate BST property (intervals are ordered)
            if (n.left) |left| {
                std.debug.assert(left.interval.last < n.interval.first);
            }
            if (n.right) |right| {
                std.debug.assert(n.interval.last < right.interval.first);
            }

            // Recurse
            self.debugValidateTree(n.left);
            self.debugValidateTree(n.right);
        }

        const NodePool = std.heap.MemoryPool(Node);

        allocator: std.mem.Allocator,
        pool: NodePool = .empty,
        root: ?*Node = null,
        node_count: usize = 0,

        /// Initialize an empty tree
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        fn allocNode(self: *Self) !*Node {
            return self.pool.create(self.allocator);
        }

        fn freeNode(self: *Self, node: *Node) void {
            self.pool.destroy(node);
        }

        /// Free all nodes and the underlying memory pool
        pub fn deinit(self: *Self) void {
            self.pool.deinit(self.allocator);
            self.root = null;
            self.node_count = 0;
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
        /// This is O(log n + k) where k is the number of affected intervals
        pub fn insertRange(self: *Self, first: T, last: T) !void {
            std.debug.assert(first <= last);
            self.root = try self.insertRangeInto(self.root, first, last);
        }

        fn insertRangeInto(self: *Self, node: ?*Node, first: T, last: T) !?*Node {
            const n = node orelse {
                const new_node = try self.allocNode();
                new_node.* = .{ .interval = IntervalType.initRange(first, last) };
                self.node_count += 1;
                return new_node;
            };

            // Check if range is entirely contained
            if (n.interval.hasRange(first, last)) {
                return n;
            }

            // Check for overlap or adjacency with current interval
            const overlaps = n.interval.intersectsRange(first, last);
            // Adjacent-left: range ends just before interval starts (last + 1 == interval.first)
            const adjacent_left = last != std.math.maxInt(T) and last + 1 == n.interval.first;
            // Adjacent-right: interval ends just before range starts (interval.last + 1 == first)
            const adjacent_right = n.interval.last != std.math.maxInt(T) and n.interval.last + 1 == first;

            if (overlaps or adjacent_left or adjacent_right) {
                // Extend the current interval to encompass the range
                n.interval.first = @min(n.interval.first, first);
                n.interval.last = @max(n.interval.last, last);

                // Now we need to absorb any overlapping/adjacent intervals from children
                n.left = self.absorbOverlapping(n.left, n);
                n.right = self.absorbOverlapping(n.right, n);

                return self.rebalance(n);
            } else if (last < n.interval.first) {
                n.left = try self.insertRangeInto(n.left, first, last);
                return self.rebalance(n);
            } else {
                n.right = try self.insertRangeInto(n.right, first, last);
                return self.rebalance(n);
            }
        }

        fn absorbOverlapping(self: *Self, node: ?*Node, target: *Node) ?*Node {
            const n = node orelse return null;

            // Check if this node's interval overlaps or is adjacent to target
            const overlaps = target.interval.intersectsRange(n.interval.first, n.interval.last);
            const adjacent = target.interval.adjacent(n.interval);

            if (overlaps or adjacent) {
                // Merge this interval into target
                target.interval.first = @min(target.interval.first, n.interval.first);
                target.interval.last = @max(target.interval.last, n.interval.last);

                // Recursively check children before destroying this node
                const new_left = self.absorbOverlapping(n.left, target);
                const new_right = self.absorbOverlapping(n.right, target);

                // Merge the children
                const merged = self.mergeSubtrees(new_left, new_right);

                self.freeNode(n);
                self.node_count -= 1;

                // Continue absorbing from merged subtree
                return self.absorbOverlapping(merged, target);
            } else if (target.interval.last < n.interval.first) {
                // Target is entirely left of this node, check left subtree
                n.left = self.absorbOverlapping(n.left, target);
                return self.rebalance(n);
            } else {
                // Target is entirely right of this node, check right subtree
                n.right = self.absorbOverlapping(n.right, target);
                return self.rebalance(n);
            }
        }

        fn mergeSubtrees(self: *Self, left: ?*Node, right: ?*Node) ?*Node {
            return self.mergeChildrenBalanced(left, right);
        }

        fn insertInto(self: *Self, node: ?*Node, elem: T) !?*Node {
            const n = node orelse {
                const new_node = try self.allocNode();
                new_node.* = .{ .interval = IntervalType.init(elem) };
                self.node_count += 1;
                return new_node;
            };

            if (n.interval.has(elem)) {
                return n;
            } else if (n.interval.adjacentLeft(elem)) {
                n.interval.first = elem;
                return self.joinLeft(n);
            } else if (n.interval.adjacentRight(elem)) {
                n.interval.last = elem;
                return self.joinRight(n);
            } else if (elem < n.interval.first) {
                n.left = try self.insertInto(n.left, elem);
                return self.rebalance(n);
            } else {
                n.right = try self.insertInto(n.right, elem);
                return self.rebalance(n);
            }
        }

        /// Delete an element from the tree
        pub fn delete(self: *Self, elem: T) void {
            self.root = self.deleteFrom(self.root, elem);
        }

        /// Delete a range of elements [first, last] from the tree
        /// This is O(log n + k) where k is the number of affected intervals
        pub fn deleteRange(self: *Self, first: T, last: T) void {
            std.debug.assert(first <= last);
            self.root = self.deleteRangeFrom(self.root, first, last);
        }

        fn deleteRangeFrom(self: *Self, node: ?*Node, first: T, last: T) ?*Node {
            const n = node orelse return null;

            // If range is entirely to the left, recurse left
            if (last < n.interval.first) {
                n.left = self.deleteRangeFrom(n.left, first, last);
                return self.rebalance(n);
            }

            // If range is entirely to the right, recurse right
            if (first > n.interval.last) {
                n.right = self.deleteRangeFrom(n.right, first, last);
                return self.rebalance(n);
            }

            // Range overlaps with this interval
            // First, handle any portion of the delete range in children
            // Guard against underflow/overflow at type boundaries
            if (first < n.interval.first and n.interval.first != std.math.minInt(T)) {
                n.left = self.deleteRangeFrom(n.left, first, n.interval.first - 1);
            }
            if (last > n.interval.last and n.interval.last != std.math.maxInt(T)) {
                n.right = self.deleteRangeFrom(n.right, n.interval.last + 1, last);
            }

            // Now handle overlap with this node's interval
            if (first <= n.interval.first and last >= n.interval.last) {
                // Entire interval is deleted - remove this node
                const result = self.mergeSubtrees(n.left, n.right);
                self.freeNode(n);
                self.node_count -= 1;
                return result;
            } else if (first <= n.interval.first) {
                // Trim from left: delete [interval.first, last], keep [last+1, interval.last]
                n.interval.first = last + 1;
                return self.rebalance(n);
            } else if (last >= n.interval.last) {
                // Trim from right: delete [first, interval.last], keep [interval.first, first-1]
                n.interval.last = first - 1;
                return self.rebalance(n);
            } else {
                // Split: keep [interval.first, first-1] and [last+1, interval.last]
                const original_last = n.interval.last;
                const right_interval = IntervalType.initRange(last + 1, original_last);
                n.interval.last = first - 1;

                // Insert the right portion into the right subtree
                const new_node = self.allocNode() catch {
                    // Restore the original interval on allocation failure
                    n.interval.last = original_last;
                    return n;
                };
                new_node.* = .{ .interval = right_interval, .left = null, .right = n.right };
                n.right = new_node;
                self.node_count += 1;
                return self.rebalance(n);
            }
        }

        fn deleteFrom(self: *Self, node: ?*Node, elem: T) ?*Node {
            const n = node orelse return null;

            if (elem < n.interval.first) {
                n.left = self.deleteFrom(n.left, elem);
                return self.rebalance(n);
            } else if (elem > n.interval.last) {
                n.right = self.deleteFrom(n.right, elem);
                return self.rebalance(n);
            } else {
                // Element is within this interval
                if (n.interval.first == n.interval.last) {
                    // Single element interval - remove the node
                    const result = self.mergeChildrenBalanced(n.left, n.right);
                    self.freeNode(n);
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
                    const original_last = n.interval.last;
                    const new_interval = IntervalType.initRange(elem + 1, original_last);
                    n.interval.last = elem - 1;

                    const new_node = self.allocNode() catch {
                        // Restore the original interval on allocation failure
                        n.interval.last = original_last;
                        return n;
                    };
                    new_node.* = .{ .interval = new_interval, .left = null, .right = n.right };
                    n.right = new_node;
                    self.node_count += 1;
                    return self.rebalance(n);
                }
            }
        }

        /// Merge two subtrees, maintaining AVL balance
        fn mergeChildrenBalanced(self: *Self, left: ?*Node, right: ?*Node) ?*Node {
            if (left == null) return right;
            if (right == null) return left;

            // Extract the rightmost node of left subtree to use as new root
            const result = self.extractRightmost(left.?);
            const new_root = result.extracted;
            const new_left = result.remaining;

            new_root.left = new_left;
            new_root.right = right;
            return self.rebalance(new_root);
        }

        const ExtractResult = struct {
            extracted: *Node,
            remaining: ?*Node,
        };

        fn extractRightmost(self: *Self, node: *Node) ExtractResult {
            if (node.right == null) {
                // This is the rightmost node - extract it
                return .{
                    .extracted = node,
                    .remaining = node.left,
                };
            }
            const result = self.extractRightmost(node.right.?);
            node.right = result.remaining;
            return .{
                .extracted = result.extracted,
                .remaining = self.rebalance(node),
            };
        }

        /// Try to merge with the rightmost node in the left subtree if adjacent
        fn joinLeft(self: *Self, node: *Node) *Node {
            if (node.left == null) return self.rebalance(node);

            // Find the rightmost node in left subtree
            const rightmost = self.findRightmost(node.left.?);

            if (rightmost.interval.last != std.math.maxInt(T) and
                rightmost.interval.last + 1 == node.interval.first)
            {
                // Merge: extend node's interval and remove the rightmost
                node.interval.first = rightmost.interval.first;
                node.left = self.removeRightmost(node.left.?);
            }

            return self.rebalance(node);
        }

        /// Try to merge with the leftmost node in the right subtree if adjacent
        fn joinRight(self: *Self, node: *Node) *Node {
            if (node.right == null) return self.rebalance(node);

            // Find the leftmost node in right subtree
            const leftmost = self.findLeftmost(node.right.?);

            if (node.interval.last != std.math.maxInt(T) and
                node.interval.last + 1 == leftmost.interval.first)
            {
                // Merge: extend node's interval and remove the leftmost
                node.interval.last = leftmost.interval.last;
                node.right = self.removeLeftmost(node.right.?);
            }

            return self.rebalance(node);
        }

        fn findRightmost(self: *Self, node: *Node) *Node {
            _ = self;
            var current = node;
            while (current.right) |right| {
                current = right;
            }
            return current;
        }

        fn findLeftmost(self: *Self, node: *Node) *Node {
            _ = self;
            var current = node;
            while (current.left) |left| {
                current = left;
            }
            return current;
        }

        fn removeRightmost(self: *Self, node: *Node) ?*Node {
            if (node.right == null) {
                // This is the rightmost node
                const left = node.left;
                self.freeNode(node);
                self.node_count -= 1;
                return left;
            }
            node.right = self.removeRightmost(node.right.?);
            return self.rebalance(node);
        }

        fn removeLeftmost(self: *Self, node: *Node) ?*Node {
            if (node.left == null) {
                // This is the leftmost node
                const right = node.right;
                self.freeNode(node);
                self.node_count -= 1;
                return right;
            }
            node.left = self.removeLeftmost(node.left.?);
            return self.rebalance(node);
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
        /// Note: Limited to trees of depth <= 64. Debug builds will assert on overflow.
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
                            std.debug.assert(self.stack_len < self.stack.len); // Tree too deep for iterator
                            self.stack[self.stack_len] = .{ .node = left, .visited_left = false };
                            self.stack_len += 1;
                        }
                        continue;
                    }

                    const interval = top.node.interval;
                    const right = top.node.right;
                    self.stack_len -= 1;

                    if (right) |r| {
                        std.debug.assert(self.stack_len < self.stack.len); // Tree too deep for iterator
                        self.stack[self.stack_len] = .{ .node = r, .visited_left = false };
                        self.stack_len += 1;
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
        /// This is O(k log n) where k is total number of intervals
        pub fn setUnion(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !Self {
            var result = Self.init(allocator);
            errdefer result.deinit();

            // Insert all intervals from self
            var it = self.iterator();
            while (it.next()) |interval| {
                try result.insertRange(interval.first, interval.last);
            }

            // Insert all intervals from other (merging happens automatically)
            var other_it = other.iterator();
            while (other_it.next()) |interval| {
                try result.insertRange(interval.first, interval.last);
            }

            return result;
        }

        /// Create a new tree that is the intersection of this tree and another
        /// Caller owns the returned tree
        /// This is O(k + m) where k and m are interval counts (merge-join algorithm)
        pub fn setIntersection(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !Self {
            var result = Self.init(allocator);
            errdefer result.deinit();

            // Use merge-join since both iterators yield intervals in sorted order
            var it = self.iterator();
            var other_it = other.iterator();

            var interval = it.next();
            var other_interval = other_it.next();

            while (interval != null and other_interval != null) {
                const a = interval.?;
                const b = other_interval.?;

                // Check for overlap
                if (a.intersectsRange(b.first, b.last)) {
                    // Compute the intersection
                    const start = @max(a.first, b.first);
                    const end = @min(a.last, b.last);
                    try result.insertRange(start, end);
                }

                // Advance the iterator whose interval ends first
                if (a.last < b.last) {
                    interval = it.next();
                } else if (b.last < a.last) {
                    other_interval = other_it.next();
                } else {
                    // Both end at the same point, advance both
                    interval = it.next();
                    other_interval = other_it.next();
                }
            }

            return result;
        }

        /// Create a new tree that is this tree minus the other tree (difference)
        /// Caller owns the returned tree
        /// This is O(k log k + m log n) where k and m are interval counts in self and other
        pub fn setDifference(self: *const Self, other: *const Self, allocator: std.mem.Allocator) !Self {
            var result = Self.init(allocator);
            errdefer result.deinit();

            // First, copy all intervals from self
            var it = self.iterator();
            while (it.next()) |interval| {
                try result.insertRange(interval.first, interval.last);
            }

            // Then delete all intervals from other
            var other_it = other.iterator();
            while (other_it.next()) |interval| {
                result.deleteRange(interval.first, interval.last);
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

test "interval count - full range signed (overflow protection)" {
    // This test verifies the overflow fix: i8 range [-128, 127] has 256 elements
    // The old implementation would overflow: 127 - (-128) = 255 which overflows i8
    const I = Interval(i8);
    const full_range = I.initRange(-128, 127);
    try std.testing.expectEqual(@as(usize, 256), full_range.count());
}

test "interval count - full range unsigned" {
    const I = Interval(u8);
    const full_range = I.initRange(0, 255);
    try std.testing.expectEqual(@as(usize, 256), full_range.count());
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
