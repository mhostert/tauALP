# Set the compiler
CXX = g++

# Define the project directories and files
SRC = generate_taus_from_beamdump.cpp
OBJ = $(SRC:.cpp=.o)
EXE = ntaus

# Pythia8 directories
PYTHIA8 = $(HOME)/Development/MG5_aMC_v3_5_6/HEPTools/pythia8
INCDIR = $(PYTHIA8)/include
LIBDIR = $(PYTHIA8)/lib

# Compilation flags
CXXFLAGS = -I$(INCDIR) -std=c++11
LDFLAGS = -L$(LIBDIR) -lpythia8 -Wl,-rpath,$(PYTHIA8)/lib

# Build target
all: $(EXE)

# Rule to build the executable
$(EXE): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Rule to build the object file
%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

# Clean rule to remove build artifacts
clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean