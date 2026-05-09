#include <iostream>
#include <cstdint>
using namespace std;

int main(){

    // 1 byte is of 8 bits
    printf("size of int data type is %ld bits \n ", sizeof(int) * 8);
    printf("size of char data type is %ld bits \n ", sizeof(char) * 8);
    printf("size of short int data type is %ld bits \n ", sizeof(short int) * 8);
    printf("size of long int data type is %ld bits \n ", sizeof(long int) * 8);
    printf("size of long long int data type is %ld bits \n ", sizeof(long long int) * 8); // this might be different for both windows and mac
    // output :
    // size of int data type is 32 bits 
    // size of char data type is 8 bits 
    // size of short int data type is 16 bits 
    // size of long int data type is 32 bits 
    // size of long long int data type is 64 bits 
    

    return 0;
}