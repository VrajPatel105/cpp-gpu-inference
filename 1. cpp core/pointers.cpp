#include <iostream>
using namespace std;


// int main(){
//     int life;
//     life = 4;

//     int card;
//     card = 50;
//     int my_card = card;

//     // pointers

//     int *myptr;
//     myptr = &my_card;

//     my_card = *myptr;

//     printf("Address is : %p", myptr); // %p because we are using a pointer 
//     printf("\n mycard is : %d", my_card); // derefrencing the pointer

//     // output : Address is : 00000067839ff78c
//     // mycard is : 50

//     printf("\n Value of the card is %d and value for my_card is %d", card, my_card);
// }


int main(){

    int score = 100;
    int *mypointer = &score;

    printf("value of score is %d \n", score);
    printf("Value of pointer is %p \n", mypointer);

    int &another_score = score; // reference
    another_score = 200;

    printf("first value of score is %d and new value after reference another_score is %d", score, score);

    return 0;

}