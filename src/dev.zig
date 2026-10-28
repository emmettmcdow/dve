//! The `dve` the in-repo harnesses import: the public API from root.zig, plus the internals
//! that benchmarks and experiments reach into. build.zig gives it to them under the name
//! `dve` but never exports it, so a consumer only ever sees root.zig.

const root = @import("root.zig");

pub const VectorEngine = root.VectorEngine;
pub const SearchResult = root.SearchResult;
pub const NoteID = root.NoteID;
pub const Error = root.Error;
pub const embed = root.embed;
pub const llama = root.llama;
pub const Reroot = root.Reroot;

pub const vec_storage = @import("vec_storage.zig");
pub const vstore = @import("vstore.zig");
pub const codes = @import("codes.zig");
pub const vec_util = @import("vec_util.zig");
pub const note_id_map = @import("note_id_map.zig");
pub const types = @import("types.zig");
pub const util = @import("util.zig");
