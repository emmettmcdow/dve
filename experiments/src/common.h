/* Shared SETUP ONLY. Nothing here runs inside a measured loop except read_exact,
 * whose cost is a function call and a compare against a ~1us syscall -- under 0.1%.
 * The measured loops themselves live inline in each binary's main. */
#ifndef COMMON_H
#define COMMON_H

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#define PAGE 4096u

/* Bypass the unified buffer cache. Darwin only; on Linux the >RAM file is the
 * only cold mechanism, which is the honest one anyway. */
static inline void set_nocache(int fd) {
#if defined(__APPLE__)
    if (fcntl(fd, F_NOCACHE, 1) == -1)
        fprintf(stderr, "warning: F_NOCACHE failed: %s\n", strerror(errno));
#else
    (void)fd;
#endif
}

/* Ask for performance cores. Best effort; failing is not fatal. */
static inline void prefer_p_cores(void) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

/* Fixed seed => identical offsets every run, so repeated runs are comparable.
 * Three shifts and three xors, chosen over rand() to keep libc out of the loop. */
static inline uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static inline void *alloc_aligned(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, PAGE, n) != 0 || p == NULL) {
        fprintf(stderr, "posix_memalign(%zu) failed\n", n);
        exit(1);
    }
    return p;
}

static inline int open_ro(const char *path, int nocache) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (nocache) set_nocache(fd);
    return fd;
}

static inline uint64_t file_size_of(int fd) {
    off_t end = lseek(fd, 0, SEEK_END);
    if (end < 0) { perror("lseek"); exit(1); }
    return (uint64_t)end;
}

/* pread may transfer short and may be interrupted, even on a regular file. */
static inline void read_exact(int fd, void *buf, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = pread(fd, (char *)buf + done, n - done, (off_t)(off + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "pread: %s\n", strerror(errno));
            exit(1);
        }
        if (r == 0) {
            fprintf(stderr, "unexpected EOF at %llu\n", (unsigned long long)(off + done));
            exit(1);
        }
        done += (size_t)r;
    }
}

/* --nocache anywhere in argv. */
static inline int has_nocache(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--nocache") == 0) return 1;
    return 0;
}

#endif
