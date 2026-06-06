#include <iostream>
using namespace std;

int main(){

    int myarray[10] = {1,3,4,5,6,7,4,3,34,12};
    int n = size(myarray);
    // for loop
    for(int i = 0; i < n; i++){
        cout << myarray[i] << endl;
    }

    // another form 

    for(int i:myarray){
        cout << myarray[i] << endl;
    }

    // while loop
    int i = 0;
    while(i < 7){
        cout << myarray[i] << endl;
        i++;
    }
}