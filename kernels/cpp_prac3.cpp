#include <iostream>

using namespace std;

// A struct (structure) is a custom data type that groups different variables under one name.
struct DeviceInfo{
    int id;
    float clockSpeed;
    bool is_available;
};

struct Particle{
    float x, y, z;
    float velocity_x, velocity_y, velocity_z;
    float mass;
};

void PrintParticle(Particle p){
    std::cout << "Position (" 
              << p.x << ", " 
              << p.y << ", " 
              << p.z << ")" 
              << std::endl;
}

int main(){
    DeviceInfo mydevice;
    mydevice.id = 1;

    DeviceInfo* ptr = &mydevice;
    ptr->id = 2;

    Particle p1;

    p1.x = 10.0f;
    p1.y = 20.0f;
    p1.z = 1.5f;

    PrintParticle(p1);

    Particle* p = new Particle;
    p->velocity_x = 1.9f;
    std::cout<<"particle pointer velocity: "<<p->velocity_x<<std::endl;

    delete p;

    return 0;
}