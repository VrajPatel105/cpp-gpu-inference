// revisiting stl by doing some tasks : 
// 1. Reads 10 numbers from stdin into a std::vector<int>
// 2. Sorts them descending using std::sort with a lambda comparator
// 3. Counts how many are even using std::count_if with another lambda
// 4. Stores frequencies in a std::unordered_map<int, int>
// 5. Prints the map using a range-based for loop

#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
using namespace std;

class Helper {
public:
    // print vector function
    void PrintVec(vector<int> v){
        for(int i = 0; i<v.size(); i++){
        cout << v[i] << endl;
        }
    }
};

// main method : 
int main(){

// obj for the helper class
Helper h;

// ---------------------------------------------------------------------------------------------- //

// // 1. Reading 10 numbers from stdin into a std::vector<int>
    vector<int> user_input;
    // get user input 
    for(int i = 0; i<10; i++){
        int current_user_input;
        cin >> current_user_input;
        user_input.push_back(current_user_input);
    }

    cout << "printing the stored vector values that were entrerd by the user" << endl;
    // once done, print the vector
    h.PrintVec(user_input);

// ---------------------------------------------------------------------------------------------- //

// 2. Sorts them descending using std::sort with a lambda comparator
    cout << "question 2" << endl;
    sort(user_input.begin(), user_input.end(), [](int a, int b){ return a > b;});
    h.PrintVec(user_input);

// ---------------------------------------------------------------------------------------------- //

// 3. Counts how many are even using std::count_if with another lambda
    cout << "question 3" << endl;
    int even_num_count = count_if(user_input.begin(), user_input.end(), [](int num){return num % 2 == 0;});
    printf("Total even number count : %d", even_num_count);

// ---------------------------------------------------------------------------------------------- //

// 4. Stores frequencies in a std::unordered_map<int, int>
    cout << "question 4" << endl;
    unordered_map<int,int> mpp;
    for(auto &it: user_input){
        mpp[it]++;
    }
    cout << "printing the frequencies from the map" << endl;
    // print the frequencies
    for(auto &pair : mpp){
        cout << pair.first << " -> " <<  pair.second << endl;
    }
// ---------------------------------------------------------------------------------------------- //

    // 5. Prints the map using a range-based for loop
    // we already did that in the above section :) 
    for(auto &pair : mpp){
        cout << pair.first << " -> " <<  pair.second << endl;
    }
    
    return 0;
}