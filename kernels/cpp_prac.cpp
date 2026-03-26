#include <iostream>

using namespace std;

float add(float a, float b){
    float result = a + b;
    return result;
}

int main(){
    int source = 20;
    int* psource = &source;

    std::cout<<psource<<"\n";

    float result = add(5.5f, 2.2f);

    std::cout<<result<<endl;

    int x = 10;
    int y = 20;
    int* ptr = &x;
    std::cout<<"Before: "<<*ptr<<endl;
    ptr = &y;
    std::cout<<"After: "<<*ptr<<endl;

    int* myData = new int;
    int val = 500;
    *myData = val;
    std::cout<<"myData: "<<*myData<<endl;

    delete myData;

    return 0;
}