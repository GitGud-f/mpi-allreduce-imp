CXX      := mpic++
CXXFLAGS := -O3 -Wall -Wextra -std=c++17
TARGET   := mpi_allreduce_imp

# Object files list
OBJS     := main.o algorithms.o constants.o

# Default Processes
NP       ?= 4




all: $(TARGET)

# Linking Phase
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)
	@echo "Build Complete."


# Compilation Phase
%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run Target
run: $(TARGET)
	@echo "Running with $(NP) processes..."
	mpirun -n $(NP) ./$(TARGET)

# Clean Artifacts
clean:
	@echo "Cleaning..."
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean run