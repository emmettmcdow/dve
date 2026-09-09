/* K random block-aligned reads. Per-read cost is (wall time) / K.
 * K is large deliberately: log2(20M) ~ 24 reads is ~2ms, which process startup
 * would swamp, so we measure many and divide. */
#include "common.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file> <block_bytes> <count> [--nocache]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    size_t bs = (size_t)strtoull(argv[2], NULL, 10);
    uint64_t count = strtoull(argv[3], NULL, 10);
    if (bs == 0 || bs % PAGE || count == 0) {
        fprintf(stderr, "block must be a nonzero multiple of %u, count > 0\n", PAGE);
        return 2;
    }

    prefer_p_cores();
    int fd = open_ro(path, has_nocache(argc, argv));
    uint64_t size = file_size_of(fd);
    char *buf = alloc_aligned(bs);
    volatile uint64_t sink = 0;
    uint64_t nblocks = size / bs;
    if (nblocks == 0) { fprintf(stderr, "file smaller than one block\n"); return 2; }
    uint64_t seed = 0x243f6a8885a308d3ULL; /* fixed: same offsets every run */

    /* ---------- measured ---------- */
    for (uint64_t i = 0; i < count; i++) {
        uint64_t blk = xorshift64(&seed) % nblocks;
        read_exact(fd, buf, bs, blk * bs);
        sink += (uint64_t)(unsigned char)buf[0];
    }
    /* ---------- /measured ---------- */

    printf("rand bs=%zu reads=%llu span=%llu sink=%llu\n", bs,
           (unsigned long long)count, (unsigned long long)size,
           (unsigned long long)sink);
    close(fd);
    free(buf);
    return 0;
}
