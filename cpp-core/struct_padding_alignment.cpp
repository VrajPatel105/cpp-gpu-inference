#include <iostream>
struct A { char c; int i; };
struct B { int i; char c; };
int main() {
    std::cout << sizeof(A) << " " << sizeof(B) << "\n";       // 8  8 
    std::cout << alignof(int) << " " << alignof(char) << "\n"; // 4  4
}