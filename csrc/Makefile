CC      ?= cc
CFLAGS  ?= -O3 -ffast-math -march=native -funroll-loops -Wall -Wextra -std=c11 -pthread
LDFLAGS := -lm

# The objects go into the shared library too, so they are position-independent whatever CFLAGS says (`override`:
# a CFLAGS given on the command line still gets it).
override CFLAGS += -fPIC

# glibc and musl hide POSIX (clock_gettime, setenv, sysconf, mmap …) under -std=c11 unless asked; macOS shows everything.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SHLIB   = libphonon2.dylib
  SHFLAGS = -dynamiclib -install_name @rpath/libphonon2.dylib
else
  override CFLAGS += -D_GNU_SOURCE
  LDFLAGS += -lpthread
  SHLIB   = libphonon2.so
  SHFLAGS = -shared
endif

LIB_SRCS = blas.c threadpool.c tensor_ops.c encoder.c decoder.c mel.c weights.c tokens.c wav.c resample.c segment.c engine.c json.c
LIB_OBJS = $(LIB_SRCS:.c=.o)
LIB      = libphonon2.a
BIN      = phonon2

all: $(BIN) $(SHLIB)

$(BIN): main.o $(LIB)
	$(CC) $(CFLAGS) -o $@ main.o $(LIB) $(LDFLAGS)

$(LIB): $(LIB_OBJS)
	ar rcs $@ $^

# The shared library, for the Python bindings (../python/phonon2.py) and other dynamic consumers.
$(SHLIB): $(LIB_OBJS)
	$(CC) $(CFLAGS) $(SHFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c phonon2.h blas.h threadpool.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf main.o $(LIB_OBJS) $(LIB) $(SHLIB) $(BIN) bench_gemm ../examples/transcribe fuzz_obj fuzz_wav fuzz_weights fuzz_phonon2 *.dSYM

.PHONY: all clean example check fuzz fuzz_phonon2 weights

# Parser fuzzing (tests/fuzz): the library and two drivers rebuilt with AddressSanitizer + UBSan into fuzz_*/,
# then tests/fuzz/fuzz.py throws mutated WAVs, manifests, binaries and token files at them. FUZZ_ITER mutations
# per input family (default 800, a few minutes).
FUZZ_ITER ?= 800
FUZZ_CFLAGS = -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -std=c11 -pthread -fPIC
fuzz_obj/%.o: %.c phonon2.h blas.h threadpool.h
	@mkdir -p fuzz_obj
	$(CC) $(FUZZ_CFLAGS) $(if $(filter-out Darwin,$(UNAME_S)),-D_GNU_SOURCE) -c -o $@ $<
FUZZ_OBJS = $(addprefix fuzz_obj/,$(LIB_OBJS))
fuzz: $(FUZZ_OBJS)
	$(CC) $(FUZZ_CFLAGS) -I. -o fuzz_wav ../tests/fuzz/fuzz_wav.c $(FUZZ_OBJS) $(LDFLAGS)
	$(CC) $(FUZZ_CFLAGS) -I. -o fuzz_weights ../tests/fuzz/fuzz_weights.c $(FUZZ_OBJS) $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=0:abort_on_error=0 python3 ../tests/fuzz/fuzz.py --iterations $(FUZZ_ITER)

# The CLI under the sanitizers, for running odd audio by hand (not part of `fuzz`).
fuzz_phonon2: $(FUZZ_OBJS)
	$(CC) $(FUZZ_CFLAGS) -I. -o $@ main.c $(FUZZ_OBJS) $(LDFLAGS)

# The library API end to end (examples/transcribe.c).
example: ../examples/transcribe.c $(LIB)
	$(CC) $(CFLAGS) -I. -o ../examples/transcribe $< $(LIB) $(LDFLAGS)

# The converted weights: ../scripts/get_weights.sh downloads the published archive, checks its pinned sha256 and
# unpacks it into ../c_weights (convert_weights.py builds the same files from Phonon-2's release instead).
weights:
	../scripts/get_weights.sh

# Regression test on the utterances in ../tests/data (needs ../c_weights, see `weights`, and python3).
check: $(BIN) example $(SHLIB)
	PHONON2_LIB=./$(SHLIB) python3 ../tests/check.py --binary ./$(BIN) --weights ../c_weights --example ../examples/transcribe

# GEMM microbenchmark (not part of the library or the CLI).
bench_gemm: bench_gemm.c $(LIB)
	$(CC) $(CFLAGS) -o $@ $< $(LIB) $(LDFLAGS)
