#include <iostream>
#include <memory>
using namespace std;

class Myclass{
public:
    Myclass(){
        cout << "-----------constructor invoked-----------" << endl;
    }

    ~Myclass(){
        cout << "-----------Destructor invoked-----------" << endl;
    }
};

int main() {
    // unique_ptr<int> unPtr1 = make_unique<int>(25);
    // // cout << unPtr1.get() << endl; // address
    // // cout << *unPtr1 << endl; // dereferencing the pointer

    // unique_ptr<int> unPtr2 = move(unPtr1); // moving the ownership.

    // cout << unPtr2.get() << endl;
    // cout << unPtr1.get() << endl;
    // // output : 
    // // 0x1bb30d06310
    // 0

    // unique_ptr<Myclass>unPtr1 = make_unique<Myclass>(); // the unique pointer will be dealocatted when the scope ends.

    // {
    //     unique_ptr<Myclass>unPtr2 = make_unique<Myclass>();
    // } // this will end when the code reaches line 33 {end of the scope}

    // shared pointer
    // shared_ptr<Myclass>shPtr1 = make_unique<Myclass>();
    // cout << "Shared count : " << shPtr1.use_count() << endl;
    // shared_ptr<Myclass>shPtr2 = shPtr1;
    // cout << "Shared count : " << shPtr2.use_count() << endl;
    // // lets put one of them in a scope
    // {
    //     shared_ptr<Myclass>shPtr3 = shPtr1;
    //     cout << "Shared count : " << shPtr3.use_count() << endl;
    // }

    // cout << "Shared count : " << shPtr1.use_count() << endl;

    // Weak Pointer : will not keep the object alive when it's last owner leave's it's score
    weak_ptr<int> wePtr1;
    {
        shared_ptr<int>shPtr1=make_shared<int>(10);
        wePtr1 = shPtr1;
    }

    return 0;
}