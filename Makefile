# Compiler configuration
CXX      := g++-16
CXXFLAGS := -std=c++26  -O3 -Wall -Wextra -march=graniterapids -freflection -MMD -MP 
SDE      := ~/sde/sde64


# Target executable name
TARGET   := test

# Source definitions
SRCS     := $(wildcard *.cpp)
OBJS     := $(SRCS:.cpp=.o)
DEPS     := $(SRCS:.cpp=.d) 

# Default compilation rule
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -fuse-ld=lld -o $(TARGET) $(OBJS)

# Compile source files to object files
%.o: %.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(DEPS)

# Clean temporary build and binary artifacts
clean:
	rm -f $(OBJS) $(TARGET) $(DEPS)  *.tmp*

run: all
	$(SDE) -gnr  -- ./$(TARGET)


.PHONY: all clean
