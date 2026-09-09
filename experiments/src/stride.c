/* Strided scan: read `span` bytes from the tail of every `period` bytes.
 * This is vstore's open scan -- one 32-byte metadata trailer per chunk.
 * At 768xf32 the chunk is one page, so every trailer sits in its own block
 * and this should cost the same as reading the whole file. */
#include "common.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file> <period_bytes> <span_bytes> [--nocache]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    uint64_t period = strtoull(argv[2], NULL, 10);
    size_t span = (size_t)strtoull(argv[3], NULL, 10);
    if (period == 0 || span == 0 || span > period) {
        fprintf(stderr, "need 0 < span <= period\n");
        return 2;
    }

    prefer_p_cores();
    int fd = open_ro(path, has_nocache(argc, argv));
    uint64_t size = file_size_of(fd);
    char *buf = alloc_aligned(span < PAGE ? PAGE : span);
    volatile uint64_t sink = 0;
    uint64_t n = size / period;
    uint64_t tail = period - span;

    /* ---------- measured ---------- */
    for (uint64_t i = 0; i < n; i++) {
        read_exact(fd, buf, span, i * period + tail);
        sink += (uint64_t)(unsigned char)buf[0];
    }
    /* ---------- /measured ---------- */

    printf("stride period=%llu span=%zu reads=%llu sink=%llu\n",
           (unsigned long long)period, span, (unsigned long long)n,
           (unsigned long long)sink);
    close(fd);
    free(buf);
    return 0;
}
