const embed_mod = @import("embed.zig");

pub const VectorEngine = @import("vector.zig").VectorEngine;
pub const SearchResult = @import("vector.zig").SearchResult;
/// What a `SearchResult` identifies a note by. `VectorEngine.pathOfResult` reads its path.
pub const NoteID = @import("note_id_map.zig").NoteID;
pub const Error = @import("vector.zig").Error;
pub const embed = embed_mod;
/// The llama.cpp backend. `llama.enabled` reports whether it was linked in
/// (`-Dllama`); the llama embedding model errors with `LlamaNotLinked` if not.
pub const llama = @import("llama.zig");
/// How a workspace root moved, for `VectorEngine.reroot`.
pub const Reroot = @import("note_id_map.zig").Reroot;
