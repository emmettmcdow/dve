const embed_mod = @import("embed.zig");

pub const VectorEngine = @import("vector.zig").VectorEngine;
pub const SearchResult = @import("vector.zig").SearchResult;
pub const Error = @import("vector.zig").Error;
pub const embed = embed_mod;
pub const vec_storage = @import("vec_storage.zig");
pub const vstore = @import("vstore.zig");
pub const codes = @import("codes.zig");
pub const note_id_map = @import("note_id_map.zig");
/// How a workspace root moved, for `VectorEngine.reroot`.
pub const Reroot = note_id_map.Reroot;
pub const types = @import("types.zig");
pub const util = @import("util.zig");
