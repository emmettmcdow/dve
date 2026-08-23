const embed_mod = @import("embed.zig");

pub const VectorEngine = @import("vector.zig").VectorEngine;
pub const SearchResult = @import("vector.zig").SearchResult;
pub const Error = @import("vector.zig").Error;
pub const embed = embed_mod;
/// The llama.cpp backend. `llama.enabled` reports whether it was linked in
/// (`-Dllama`); the llama embedding model errors with `LlamaNotLinked` if not.
pub const llama = @import("llama.zig");
pub const vec_storage = @import("vec_storage.zig");
pub const vstore = @import("vstore.zig");
pub const codes = @import("codes.zig");
pub const vec_util = @import("vec_util.zig");
pub const note_id_map = @import("note_id_map.zig");
pub const types = @import("types.zig");
pub const util = @import("util.zig");
