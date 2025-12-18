const std = @import("std");
const dietz = @import("dietz");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tree = dietz.Tree(i32).init(allocator);
    defer tree.deinit();

    try tree.insert(1);
    try tree.insert(2);
    try tree.insert(3);
    try tree.insert(10);
    try tree.insert(11);
    try tree.insert(12);

    std.debug.print("DIET Demo\n", .{});
    std.debug.print("Inserted: 1, 2, 3, 10, 11, 12\n", .{});
    std.debug.print("Interval count: {}\n", .{tree.intervalCount()});
    std.debug.print("Element count: {}\n", .{tree.count()});

    const ivs = try tree.intervals(allocator);
    defer allocator.free(ivs);

    std.debug.print("Intervals:\n", .{});
    for (ivs) |interval| {
        std.debug.print("  [{}, {}]\n", .{ interval.first, interval.last });
    }

    std.debug.print("\nContains 2: {}\n", .{tree.contains(2)});
    std.debug.print("Contains 5: {}\n", .{tree.contains(5)});
}

test "simple test" {
    const gpa = std.testing.allocator;
    var tree = dietz.Tree(i32).init(gpa);
    defer tree.deinit();

    try tree.insert(42);
    try std.testing.expect(tree.contains(42));
}
