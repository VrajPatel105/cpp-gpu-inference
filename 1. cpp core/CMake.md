# CMake

## CMake stands for 'cross-platform make' where cmake uses a configuration file called CMakeLists.txt

    a. Define your project in CMakeLists.txt
    b. Run CMake to create the Makefile
    c. Build your project using Make
    d. Add code, fix things, etc then jump to step c.
    e. If you add a new .c files or alter the dependencies then jumps to step a


## Our of source builds
- traditionally Makefile are placed in the same directory as the source code. 
    - You go into the course directory and then run Make.

- But CMake is different. The Makefile is generated in a different directory.
    - The CMakeLists.txt file is in the source directory, but hte automatically generated build files are seperate,
        so as not to overwhelm your course directory

- /home/vraj/src/hellow
    - hellow.c
    - CMakeLists.txt
    - build/
        - Makefile
        - CMakeCache.txt
        - cmake_install.cmake
        - CmakeFiles/
        - hellow

# CMAKE Readme by AI :)

# CMake Notes

My personal reference for CMake, written while setting up the `cpp-gpu-inference` project.

## What problem CMake solves

Compiling one file by hand is fine:

```bash
g++ -std=c++20 file.cpp -o file
```

But a real project has many `.cpp` files, needs specific compiler flags, links libraries, and has to build on Windows, Mac, and Linux. Typing the right `g++` command for every file with every flag gets painful and breaks across machines.

CMake fixes that. Describe the project once in a `CMakeLists.txt`, and CMake figures out the actual build commands for whatever machine it runs on.

## The key idea: CMake is a generator

CMake does NOT compile code. It reads `CMakeLists.txt` and generates build files for a real build tool, which then does the compiling.

Two layers:

```
CMake (the generator)  ->  a real build tool (does the compiling)
```

Examples of the real build tool:
- Windows + MinGW: generates "MinGW Makefiles" for `mingw32-make`
- Mac/Linux: generates files for `make`
- Visual Studio: generates `.sln` project files

This is why a "generator not found" error happens: CMake tried to generate for a build tool that isn't installed. The fix is to point it at one that is, e.g. `-G "MinGW Makefiles"`.

## The two-step cycle

**Step 1, configure:**

```bash
cmake ..
```

Reads `CMakeLists.txt` and generates build files into the `build` folder. Run this once, and again only when `CMakeLists.txt` itself changes (added a file, changed a setting).

**Step 2, build:**

```bash
cmake --build .
```

Runs the real build tool to compile the code. Run this every time a `.cpp` file is edited.

## The build folder

All generated files live in a separate `build` folder, kept apart from source so the project folder stays clean. That is why the setup is:

```bash
mkdir build
cd build
cmake ..
```

The `build` folder is generated, not source. It goes in `.gitignore`.

## Minimal CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)   # lowest CMake version that can build this
project(cpp_core)                      # project name

set(CMAKE_CXX_STANDARD 20)             # use C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)    # fail if C++20 unavailable, don't silently downgrade

add_executable(hellow hellow.cpp)      # build hellow.exe from hellow.cpp
```

Line by line:
- `cmake_minimum_required` — the lowest CMake version allowed
- `project` — names the project
- `set(CMAKE_CXX_STANDARD 20)` — the line that permanently removes the need for `-std=c++20`
- `add_executable(name source.cpp)` — make an executable called `name` from these source files

## Adding more programs

A second program is another `add_executable` line:

```cmake
add_executable(customConcepts customConcepts.cpp)
```

A program built from multiple files:

```cmake
add_executable(myapp main.cpp helper.cpp math.cpp)
```

## Windows + MinGW setup (the working sequence)

If CMake defaults to NMake (which needs Visual Studio) but only g++/MinGW is installed, force the MinGW generator. The build folder must be deleted first if it already cached the wrong generator:

```bash
rmdir /s /q build
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
```

A successful configure shows the compiler being detected, e.g. `The CXX compiler identification is GNU 13.2.0`. A successful build ends with `Built target <name>` and produces a runnable `.exe`.

## Quick reference

| Action | Command |
| --- | --- |
| First-time configure | `cmake ..` |
| Configure with MinGW | `cmake .. -G "MinGW Makefiles"` |
| Build / rebuild | `cmake --build .` |
| Clean rebuild | delete `build` folder, then reconfigure |

Re-run `cmake ..` only when `CMakeLists.txt` changes. Re-run `cmake --build .` after every source edit.