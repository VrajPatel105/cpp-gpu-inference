#include <iostream>
#include <cstring>
using namespace std;


// rvalue references
// ----------------------------------------------------------------------------
// lvalue is an object that occupies a persistent memory location.

// for example
int x = 10; // defined in function a 

// here when the variable goes out of scope (out of function a), the memory assicoated 
        // with variable x will be deallocated

    // since x has a persistent memory location, it's LVALUE

// ----------------------------------------------------------------------------

// rvalue is a temporary value that does not have a persistent memory location. 
    // meaning that we cannot access or modify the value at all. It's only for the temp variable
    // once done, it's deallocated
int y = 10; // this is temp lets say
// if we apply the same above example, this is a RVALUE. 
// so when the line int y = 10; is completed, the memory assicoated with y will be deallocated
// since this is not a persistent memory location, it's called rvalue.


// ----------------------------------------------------------

// Lvalue reference -> binds to lvalue
// int& lr = x; -> here the & means it's lvalue reference
// int& r = 5; => 5 r is a rvalue, so this is not allowed because we cannot give lvalue reference to rvalue

// Rvalue reference -> binds to rvalue
// int && rr = 5; // rvalue reference
// significance the rvalue reference :

// rvalue reference allows to capture temporary object in the move constructor and 
    // move it's resources to another object.





// Move constructor 
// Allows stealing resources from a temporary obj, and provide those resources to some other obj.




class MyString{
    char* data;

public:
    MyString(const char* s){
        data = new char[strlen(s) + 1];
        strcpy(data,s);
        cout << "Constructed" << endl;
    }


};


int main(){
    MyString s1("Hello");
    return 0;
}