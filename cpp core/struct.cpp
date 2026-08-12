#include <iostream>
using namespace std;

struct User{
    const int uId; // this const is pointing to the value 
    const char *name; // this const is a pointer pointing to the address only
    const char *email;
    int course_count;
};

int main(){

    User vraj = {001,"VrajPatel","vraj@gmail.com",2};
    User abc = {002,"apatel","a@gmail.com",3};

    return 0;
}