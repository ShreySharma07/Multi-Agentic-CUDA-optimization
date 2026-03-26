#include <iostream>

// An int is usually 4 bytes.

// A float is usually 4 bytes.

// sizeof(float) tells the compiler to calculate this for you automatically.

int main(){
    // int N = 100;
    // float* N_array = new float[N];

    //Exercise 2.1
    int N = 10;
    double total_size = N*sizeof(double);
    std::cout<<total_size<<std::endl;

    //Exercise 2.2
    int n = 5;
    float* arr = new float[n];

    for(int i=0; i<n; i++){
        arr[i] = i*1.1f;
    }

    for(int i=0; i<n; i++){
        std::cout<<arr[i]<<std::endl;
    }

    delete[] arr;

    //Exercise 2.3
    int ar[] = {10, 20, 30};
    int* p = ar;
    std::cout<<"p value: "<<*p<<std::endl;

    p++;
    std::cout<<"p value after increment: "<<*p<<std::endl;


    //Modern Way of doing this here we do not need to do memory management and delete
    int m = 5;
    std::vector<float> a(n);
    
    for(int i=0; i<a.size(); i++){
        a[i] = i * 1.1f;
    }

    for(int i = 0; i < arr.size(); i++){
        std::cout << arr[i] << std::endl;
    }
}