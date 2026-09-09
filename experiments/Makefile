CC     ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=gnu11
BINS   := mkdata seq stride rand

all: $(BINS)

$(BINS): %: src/%.c src/common.h
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

# Data files are large (up to 32 GiB); removing them is separate from `clean`
# so a rebuild does not silently cost half an hour of regeneration.
clean-data:
	rm -f *.dat

.PHONY: all clean clean-data
