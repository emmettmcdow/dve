const embed_mod = @import("embed.zig");
const config = @import("config");

/// The embedding model configured at build time.
pub const embedding_model: embed_mod.EmbeddingModel =
    @enumFromInt(@intFromEnum(config.embedding_model));

pub const VectorDB = @import("vector.zig").VectorDB;
pub const SearchResult = @import("vector.zig").SearchResult;
pub const Error = @import("vector.zig").Error;
pub const embed = embed_mod;
pub const vec_storage = @import("vec_storage.zig");
pub const note_id_map = @import("note_id_map.zig");
pub const types = @import("types.zig");
pub const util = @import("util.zig");
