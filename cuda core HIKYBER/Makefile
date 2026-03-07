NVCC      = nvcc
ARCH      = sm_86
STD       = c++17
CUDA_FLAGS = --std=$(STD) -arch=$(ARCH) -rdc=true -lineinfo
OPT_FLAGS  = -O3 --use_fast_math
DBG_FLAGS  = -G -g -DDEBUG
INCLUDES   = -Iinclude

# Add -DUSE_PLANTARD to switch modular reduction suite
# EXTRA_FLAGS = -DUSE_PLANTARD

SRC_DIR   = src
INC_DIR   = include
TEST_DIR  = tests
BUILD_DIR = build

SRCS = $(SRC_DIR)/main.cu
TARGET = $(BUILD_DIR)/hi_kyber

KAT_SRCS = $(TEST_DIR)/kat_test.cu
KAT_TARGET = $(BUILD_DIR)/kat_test

.PHONY: all release debug clean kat

all: release

release: $(TARGET)

$(TARGET): $(SRCS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(OPT_FLAGS) $(INCLUDES) $(EXTRA_FLAGS) -o $@ $<

debug: | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(DBG_FLAGS) $(INCLUDES) $(EXTRA_FLAGS) -o $(TARGET)_dbg $(SRCS)

kat: $(KAT_TARGET)

$(KAT_TARGET): $(KAT_SRCS) | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(OPT_FLAGS) $(INCLUDES) $(EXTRA_FLAGS) -o $@ $<

$(BUILD_DIR):
	mkdir $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

# Show register usage
reginfo: | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(OPT_FLAGS) $(INCLUDES) $(EXTRA_FLAGS) --ptxas-options=-v -o $(TARGET) $(SRCS)
