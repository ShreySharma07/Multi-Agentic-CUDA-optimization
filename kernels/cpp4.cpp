#include <iostream>
#include <stdio.h>

using namespace std;

/*3. Templates (Generic Programming)
This is vital for "Tele-Op" (your project). 
You don't want to write one CUDA kernel for int and another for float. 
You write one Template that works for both.
typename and class mean the same thing in templates:*/

template <typename T>
T getMax(T a, T b){
    return (a > b) ? a : b;
}

template <typename T>
T getMul(T a, T b){
    return (a * b);
}

struct Particle{
    float x, y, z;
    float velocity_x, velocity_y, velocity_z;
    float mass;
};

int main(){
    auto speed = 5.5f;
    auto myParticle = Particle();

    /*2. Lambdas (Anonymous Functions)
    A lambda is a function you define "on the fly" inside another function. 
    In modern CUDA libraries like Thrust, you use these to tell the GPU how 
    to sort or transform data without writing a whole separate function.*/
    auto add = [](float a, float b){return a + b;};
    float sum = add(10.0f, 5.0f);

    float f = getMax(10.0f, 5.0f);
    std::cout<<"Max of both numbers are: "<<f<<endl;

    float multiply = getMul(34.0f, 4.0f);
    std::cout<<"Multiple of 2 numbers is: "<<multiply<<endl;

    auto checkLimit = [](float a){
        if (a>100.0f){
            std::cout<<"Limit Exceeded!"<<endl;
        }
        else{
            std::cout<<"Within Limit!"<<endl;
        }
    };
    checkLimit(multiply);

    return 0;
}