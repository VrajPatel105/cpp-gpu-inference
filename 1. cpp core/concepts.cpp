// Concepts are an extension to the 'templates' feature that is introduced in c++ 20. 
// Concepts are named Boolean predicates on template parameters, evaluated at compile time.
// A concept may be associated with a template (class template, func template, member func of a class template), in which case it serves as a constraint
// it limits the set of arguments that are accepted as template parameters.

#include <iostream>
#include <concepts> // a compile time 'predicate' (test) on our time to make sure that we can use our templated function
using namespace std;

template<typename T>
void Print(T value){
    cout << "The value is : " << value << endl; 
}

template <typename T>
bool IsEqual(T a, T b){
    return a == b;
}

int main(){
    Print(5);
    cout << "1==1:" << boolalpha << IsEqual(1,1) << endl;
    return 0;
}