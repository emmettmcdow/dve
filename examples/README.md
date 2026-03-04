# Examples

Each example implements the same `embed-watch` CLI with two subcommands:

- `watch <dir>` - Watches a directory for changes to `.md` files and re-embeds them as they change.
- `search <dir> <query>` - Searches the embedded database in `<dir>` and prints results ranked by similarity.

| Example | Status |
|---------|--------|
| [Zig](./zig) | Available |
| [Swift](./swift) | Coming soon |

## Zig
Does this change things?

```sh
cd zig
zig build
./zig-out/bin/embed-watch watch /path/to/docs
./zig-out/bin/embed-watch search /path/to/docs "my query"
```

Or with `zig build run`:
```sh
zig build run -- watch /path/to/docs
zig build run -- search /path/to/docs "my query"
```
