#include <iostream>

using namespace std;

int main(){
    int source = 20;
    int* psource = &source;

    std::cout<<psource<<"\n";

    return 0;
}