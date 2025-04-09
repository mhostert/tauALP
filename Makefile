# Set the compiler
CXX = g++

# Define the project directories and files
SRC1 = generate_taus_from_beamdump.cpp
OBJ1 = $(SRC1:.cpp=.o)
EXE1 = ntaus

SRC2 = generate_proton_events.cpp
OBJ2 = $(SRC2:.cpp=.o)
EXE2 = nprotons

# Pythia8 directories
PYTHIA8 = $(HOME)/Development/MG5_aMC_v3_5_6/HEPTools/pythia8
INCDIR = $(PYTHIA8)/include
LIBDIR = $(PYTHIA8)/lib

# Compilation flags
CXXFLAGS = -I$(INCDIR) -std=c++11
LDFLAGS = -L$(LIBDIR) -lpythia8 -Wl,-rpath,$(PYTHIA8)/lib

# Build target
all: $(EXE1) $(EXE2)

# Rule to build the executable
$(EXE1): $(OBJ1)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Rule to build the executable
$(EXE2): $(OBJ2)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Rule to build the object file
%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

# Clean rule to remove build artifacts
clean:
	rm -f $(OBJ) $(EXE)

.PHONY: all clean