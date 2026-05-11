# C++ Core

Revisiting C++ from scratch as part of my ML systems roadmap. This folder is the W1 + W2 foundation work before moving into GPU kernels, CUDA, and the transformer C++ port.

## What's in here

Small standalone files, each focused on one concept. Compiled and run individually.

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
- **move_semantics.cpp** — custom `String` class with constructor and destructor, deep copy issue setup, Lvalue vs Rvalue.

## Topics covered

- I/O: `cin`, `cout`, `getline`, `printf`
- Primitive types and their sizes
- Arrays and iteration
- Functions and headers
- Pointers and references
- Structs and `const` qualifiers
- Smart pointers: `unique_ptr`, `shared_ptr`, `weak_ptr`
- Move semantics, Lvalue and Rvalue, `std::move`
- Perfect forwarding and universal references
- Function templates and class templates
- C++20 Concepts
- STL: `vector`, `unordered_map`, `<algorithm>`, lambdas as comparators
- Memory layout and alignment (`alignof`, `alignas`, struct padding)
- CMake basics for multi-file builds

## Compiling

```bash
g++ -std=c++20 filename.cpp -o filename
./filename
```