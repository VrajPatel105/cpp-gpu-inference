#include <iostream>
#include <memory>
using namespace std;

int main() {
    unique_ptr<int> unPtr1 = make_unique<int>(25);
    // cout << unPtr1.get() << endl; // address
    // cout << *unPtr1 << endl; // dereferencing the pointer

    unique_ptr<int> unPtr2 = move(unPtr1); // moving the ownership.

    cout << unPtr2.get() << endl;
    cout << unPtr1.get() << endl;
    // output : 
    // 0x1bb30d06310
    // 0

    

    return 0;
}