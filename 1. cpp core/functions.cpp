#include <iostream>
using namespace std;

void sayHello(){
    puts("Hello vraj");
}

int addNums(int a, int b){
    int sum = a + b;
    return sum;
}

int main(){

    sayHello();

    int x = 3;
    int y = 5;

    int answer = addNums(x,y);
    printf("our final answre %d" , answer);
    return 0;
}