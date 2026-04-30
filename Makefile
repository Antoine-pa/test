CC      = gcc
SRC     = matrixMultiply.c
BIN     = matrixMultiply

# ── max barbarism ──────────────────────────────────────────────
#  -O3                  full optimisation pass
#  -march=native        use every ISA extension the CPU has (AVX2, FMA…)
#  -mtune=native        schedule µ-ops for this exact micro-arch
#  -ffast-math          reorder / contract FP ops (enables FMA contraction)
#  -funroll-loops       unroll inner loops for ILP
#  -ftree-vectorize     (implied by -O3, explicit for clarity)
#  -flto                link-time optimisation across TU boundaries
#  -fopenmp             OpenMP threading
#  -mavx2 -mfma         belt-and-suspenders SIMD flags
CFLAGS  = -O3 -march=native -mtune=native \
          -ffast-math -funroll-loops -ftree-vectorize \
          -flto -fopenmp -mavx2 -mfma -DNDEBUG

LDFLAGS = -flto -fopenmp -lm

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

debug:
	$(CC) -O0 -g -fopenmp $(SRC) -o $(BIN)

clean:
	rm -f $(BIN)

.PHONY: all debug clean
