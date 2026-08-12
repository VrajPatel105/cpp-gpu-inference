# C++ Core

Revisiting C++ from scratch as part of my ML systems roadmap. This folder is the W1 + W2 foundation work before moving into GPU kernels, CUDA, and the transformer C++ port.

## What's in here

Small standalone files, each focused on one concept. Compiled and run individually, or built together via CMake.

### Files

- **basic.cpp** — first program. `cin`, `cout`, `getline`, basic string I/O.
- **integers.cpp** — integer sizes in bits using `sizeof`. Covers `int`, `char`, `short`, `long`, `long long`.
- **arrays.cpp** — declaring arrays with and without initializers, indexing, default values.
- **iteration.cpp** — for loop, range-based for loop, while loop over an array.
- **functions.cpp** — function declaration, return types, void functions, passing parameters.
- **header.cpp** + **adder.h** — splitting code across files using a header. Include guards with `#ifndef`.
- **pointers.cpp** — pointers (`*`, `&`), dereferencing, address printing with `%p`, and references as aliases to existing variables.
- **struct.cpp** — structs with `const` fields, distinction between `const int` (value is const) and `const char*` (pointer to const data).
- **smartpointers.cpp** — `unique_ptr`, `shared_ptr`, `weak_ptr` from `<memory>`. Covers `make_unique`, `make_shared`, ownership transfer with `std::move`, reference counting with `use_count()`, and scope-based deallocation.
- **move_semantics.cpp** — custom `String` class with constructor and destructor, deep copy setup, Lvalue vs Rvalue.
- **move_constructor.cpp** — implementing a move constructor, stealing resources from a temporary instead of deep copying.
- **concepts.cpp** — C++20 Concepts intro. Why concepts exist: constrained template parameters and readable errors.
- **customConcepts.cpp** — writing custom concepts. `Numeric` built from `std::integral` / `std::floating_point`, `Addable` using a `requires` expression, attaching concepts to templates.
- **stl.cpp** — STL practice. `vector`, `unordered_map`, `<algorithm>` (`sort`, `count_if`), lambdas as comparators.
- **struct_padding_alignment.cpp** — struct padding, `sizeof`, `alignof`. Why member order changes struct size.

### Build files

- **CMakeLists.txt** — CMake build config. Sets C++20 standard, defines executable targets.
- **CMake.md** — personal notes on how CMake works.

## Topics covered

- I/O: `cin`, `cout`, `getline`, `printf`
- Primitive types and their sizes
- Arrays and iteration
- Functions and headers
- Pointers and references
- Structs and `const` qualifiers
- Smart pointers: `unique_ptr`, `shared_ptr`, `weak_ptr`, and RAII
- Move semantics, Lvalue and Rvalue, `std::move`, move constructor
- Perfect forwarding and universal references
- Function templates and class templates
- C++20 Concepts and the `requires` expression
- STL: `vector`, `unordered_map`, `<algorithm>`, lambdas as comparators
- Memory layout and alignment (`alignof`, `alignas`, struct padding)
- CMake for multi-file builds

## Compiling

Single file:

```bash
g++ -std=c++20 filename.cpp -o filename
./filename
```

With CMake:

```bash
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
```