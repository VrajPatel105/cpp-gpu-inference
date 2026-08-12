#include <iostream>
using namespace std;

void sayHello(){
    puts("Hello vraj");
}

int addNums(int a, int b){
    int sum = a + b;
    return sum;
}

int main(){

    sayHello();

    int x = 3;
    int y = 5;

    int answer = addNums(x,y);
    printf("our final answre %d" , answer);
    return 0;
}

/** 
# C++20 Concepts

## The problem they solve

Before concepts, C++ templates accepted any type:

```cpp
template <typename T>
T add(T a, T b) { return a + b; }
```

Two issues with this:

1. **Bad error messages.** If you accidentally call `add` with a type that doesn't support `+` (like `std::vector`), the compiler dumps a 50-line error buried deep inside standard library headers, not at the call site.

2. **Silent signatures.** The function signature doesn't tell you what types are valid. You have to read the body and infer the requirements.

## What concepts do

A concept is a named, reusable constraint on what a type must support. Attach it to a template parameter and the compiler checks it BEFORE trying to instantiate the template.

```cpp
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
T add(T a, T b) { return a + b; }
```

Now:
- The signature self-documents: "T must be Numeric."
- If someone calls `add` with a vector, the error is one clean line at the call site: "constraint Numeric not satisfied for std::vector<int>."

## Mental model

A concept is an interface for a TYPE, not an object. It answers "what does this type need to support?" If the type satisfies the requirements, the template compiles. If not, you get a readable compile-time error.

Closest analogy: Python type hints, but actually enforced by the compiler.

## Key syntax

**Defining a concept:**
```cpp
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;
```

**Using a concept on a template:**
```cpp
template <Numeric T>
T add(T a, T b) { return a + b; }
```

**Combining concepts with `&&` and `||`:**
```cpp
template <typename T>
concept Printable = Numeric<T> && requires(T x) { std::cout << x; };
```

**Using `requires` to express custom requirements:**
```cpp
template <typename T>
concept Addable = requires(T a, T b) {
    a + b;
};
```

## Why this matters

Concepts make template code:
- **Readable** — requirements live in the signature, not hidden in the body
- **Safer** — invalid types are rejected early with clear errors
- **Composable** — concepts combine with `&&` and `||` to build more specific constraints

Modern C++ libraries (PyTorch ATen, Eigen, standard ranges) use concepts extensively, so reading modern C++ code requires understanding them.

**/