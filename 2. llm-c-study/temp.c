#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>


int main(){

    int param_sizes[16] = {
    38633472,
    786432,
    9216,
    9216,
    21233664,
    27648,
    7077888,
    9216,
    9216,
    9216,
    28311552,
    36864,
    28311552,
    9216,
    768,
    768
};
 // this is only to understand the malloc_and_point_parameters function 
    size_t num_parameters = 0;
    for (size_t i = 0; i < 16; i++) {
        printf("At i=%d , num_parameters = %d \n", i, num_parameters);
        num_parameters += param_sizes[i];
        }

    // the outp8ut for above loop : 
// so far at i=0 , num_parameters = 0 
// so far at i=1 , num_parameters = 38633472 
// so far at i=2 , num_parameters = 39419904 
// so far at i=3 , num_parameters = 39429120 
// so far at i=4 , num_parameters = 39438336 
// so far at i=5 , num_parameters = 60672000 
// so far at i=6 , num_parameters = 60699648 
// so far at i=7 , num_parameters = 67777536 
// so far at i=8 , num_parameters = 67786752 
// so far at i=9 , num_parameters = 67795968 
// so far at i=10 , num_parameters = 67805184 
// so far at i=11 , num_parameters = 96116736 
// so far at i=12 , num_parameters = 96153600 
// so far at i=13 , num_parameters = 124465152 
// so far at i=14 , num_parameters = 124474368 
// so far at i=15 , num_parameters = 124475136 

    return 0;
}