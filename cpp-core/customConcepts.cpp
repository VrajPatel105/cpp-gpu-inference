// writing the custom code for concepts
#include <iostream>
#include <concepts>

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <typename T>
concept Addable = requires(T a, T b) { a + b; };

template <Numeric T>
T add(T a, T b) { return a + b; }

int main() {
    std::cout << add(3, 4) << "\n";       // works, int is Numeric
    std::cout << add(2.5, 1.5) << "\n";   // works, double is Numeric
    // std::cout << add("hi", "bye");     // uncomment this — read the error
    return 0;
}