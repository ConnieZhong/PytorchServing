# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/conniezhong/cpp/PytorchServing/third_party/cpptoml

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/cpptoml-conversions.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/cpptoml-conversions.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/cpptoml-conversions.dir/flags.make

examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o: examples/CMakeFiles/cpptoml-conversions.dir/flags.make
examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o: ../examples/conversions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o"
	cd /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o -c /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/examples/conversions.cpp

examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpptoml-conversions.dir/conversions.cpp.i"
	cd /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/examples/conversions.cpp > CMakeFiles/cpptoml-conversions.dir/conversions.cpp.i

examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpptoml-conversions.dir/conversions.cpp.s"
	cd /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/examples/conversions.cpp -o CMakeFiles/cpptoml-conversions.dir/conversions.cpp.s

examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o.requires:

.PHONY : examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o.requires

examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o.provides: examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/cpptoml-conversions.dir/build.make examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o.provides.build
.PHONY : examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o.provides

examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o.provides.build: examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o


# Object files for target cpptoml-conversions
cpptoml__conversions_OBJECTS = \
"CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o"

# External object files for target cpptoml-conversions
cpptoml__conversions_EXTERNAL_OBJECTS =

cpptoml-conversions: examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o
cpptoml-conversions: examples/CMakeFiles/cpptoml-conversions.dir/build.make
cpptoml-conversions: examples/CMakeFiles/cpptoml-conversions.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../cpptoml-conversions"
	cd /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpptoml-conversions.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/cpptoml-conversions.dir/build: cpptoml-conversions

.PHONY : examples/CMakeFiles/cpptoml-conversions.dir/build

examples/CMakeFiles/cpptoml-conversions.dir/requires: examples/CMakeFiles/cpptoml-conversions.dir/conversions.cpp.o.requires

.PHONY : examples/CMakeFiles/cpptoml-conversions.dir/requires

examples/CMakeFiles/cpptoml-conversions.dir/clean:
	cd /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/cpptoml-conversions.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/cpptoml-conversions.dir/clean

examples/CMakeFiles/cpptoml-conversions.dir/depend:
	cd /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/conniezhong/cpp/PytorchServing/third_party/cpptoml /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/examples /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/examples /home/conniezhong/cpp/PytorchServing/third_party/cpptoml/build/examples/CMakeFiles/cpptoml-conversions.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/cpptoml-conversions.dir/depend

