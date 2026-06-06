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
    // copy constructor
    MyString(const MyString& other){ // since we s1 is lvalue, we define other as lvalue reference here.
        data = new char[strlen(other.data) + 1];
        strcpy(data, other.data);
        cout << "Copied " << endl;
    }

    // defining the destructors 
    ~MyString(){
        delete[] data;
        cout << "Destroyed" << endl;
    }

    // void print(){ cout << data << endl;}

    MyString(MyString&& other){ // this is the entire rvalue object 
        // && means that it's a rvalue reference -> temporary 
        data = other.data; // getting the data from s1 to s2 : both pointers are pointing to the same string's array in the memory
        other.data = nullptr; // pointing the other.data (s1) to null pointer now.
        // so now we end up with only s2 data in the memory 
        cout << "Moved" << endl;
    }

    // updating the print function
    void print() {
        if (data){
            cout << data << endl;
        }
        else{
            cout << "Empty" << endl; // this will only print if the obj is pointing to nullptr
        }
    }

};


int main(){
    // MyString s1("Hello");
    // // copying object
    // MyString s2 = s1; // here since s1 is lvalue (because we are storing the string in an array) meaning that it has a persistent value
    
    // s1.print();
    // s2.print();
    // output : 
    // Constructed
    // Copied 
    // Hello
    // Hello
    // Destroyed
    // Destroyed
    
    // but here above, we used copy constructor which actually made 2 arrays in the memory
    // and we do not want that
    // therefore, we use the move constructor
    // move constructor -> we want to transfer the ownership from one object to another

    // and in order to do this, we need to tell the compiler that consider s1 should be treated as temporary object
    // therefore; 
    MyString s1("Hello");
    MyString s2 = move(s1); // here this means s1 is treated as temporary value 
    // which means that s1 is a rvalue. and because it's an rvalue, we now need a rvalue reference in the constructor as well
    s1.print();
    s2.print();

    // above output : 
    // Constructed
    // Moved
    // Empty
    // Hello
    // Destroyed
    // Destroyed

    return 0;
}