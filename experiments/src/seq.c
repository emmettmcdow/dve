/* Sequential full-file scan: consecutive preads from 0 to EOF.
 * This is what "examine every vector" costs. */
#include "common.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <file> <block_bytes> [--nocache]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    size_t bs = (size_t)strtoull(argv[2], NULL, 10);
    if (bs == 0 || bs % PAGE) {
        fprintf(stderr, "block must be a nonzero multiple of %u\n", PAGE);
        return 2;
    }

    prefer_p_cores();
    int fd = open_ro(path, has_nocache(argc, argv));
    uint64_t size = file_size_of(fd);
    /* The loop reads whole blocks only, so a block larger than the file would
     * time zero reads and report it as a very fast scan. */
    if ((uint64_t)bs > size) {
        fprintf(stderr, "block %zu exceeds file size %llu\n", bs,
                (unsigned long long)size);
        return 2;
    }
    char *buf = alloc_aligned(bs);
    volatile uint64_t sink = 0;
    uint64_t off = 0;

    /* ---------- measured ---------- */
    while (off + bs <= size) {
        read_exact(fd, buf, bs, off);
        sink += (uint64_t)(unsigned char)buf[0];
        off += bs;
    }
    /* ---------- /measured ---------- */

    printf("seq bs=%zu read=%llu reads=%llu sink=%llu\n", bs,
           (unsigned long long)off, (unsigned long long)(off / bs),
           (unsigned long long)sink);
    close(fd);
    free(buf);
    return 0;
}
