/* Generate a test file of a given size. Skips regeneration when the file is
 * already exactly the requested size, so a 32 GiB run is paid for once. */
#include "common.h"
#include <sys/stat.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <file> <size_bytes>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    uint64_t want = strtoull(argv[2], NULL, 10);

    struct stat st;
    if (stat(path, &st) == 0 && (uint64_t)st.st_size == want) {
        printf("mkdata %s already %llu bytes, skipping\n", path,
               (unsigned long long)want);
        return 0;
    }

    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }

    size_t chunk = 1u << 20;
    unsigned char *buf = alloc_aligned(chunk);
    /* Nonzero and varying, so nothing downstream can special-case a hole. */
    for (size_t i = 0; i < chunk; i++) buf[i] = (unsigned char)(i * 31u + 7u);

    uint64_t off = 0;
    while (off < want) {
        size_t n = (want - off) < chunk ? (size_t)(want - off) : chunk;
        size_t done = 0;
        while (done < n) {
            ssize_t w = pwrite(fd, buf + done, n - done, (off_t)(off + done));
            if (w < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "pwrite: %s\n", strerror(errno));
                return 1;
            }
            done += (size_t)w;
        }
        off += n;
    }
    if (fsync(fd) != 0) { perror("fsync"); return 1; }
    close(fd);
    free(buf);
    printf("mkdata %s %llu bytes\n", path, (unsigned long long)want);
    return 0;
}
