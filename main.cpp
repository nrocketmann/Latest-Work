//
// Created by Nameer Hirschkind on 5/12/20.
//
#include "Drone.h"
//#include <pybind11/pybind11.h>

Drone* make_simple_drone() {

    double mass = 2;
    double max_force = 5;
    double posx = 0;
    double posy = 0;
    double posz = 0;
    double velx = 0;
    double vely = 0;
    double velz = 0;
    double angx = 0;
    double angy = 0;
    double angz = 0;
    double r_torque = 1;
    double orientationx = 1;
    double orientationy = 0;
    double orientationz = 0;
    double forwardx = 0;
    double forwardy = 1;
    double forwardz = 0;
    double arm = .3;


    Drone* d = new Drone(mass,max_force,posx,posy,posz,velx,vely,velz,angx,angy,angz,r_torque,orientationx,orientationy,
            orientationz,forwardx,forwardy,forwardz,arm);
    return d;
}

Drone* make_harder_drone() {
    double mass = 5;
    double max_force = 10;
    Vector3d pos;
    pos<<0,0,0;
    Vector3d vel;
    vel<<0,0,0;
    Vector3d ang;
    ang<<0,0,0;
    double r_torque = 1;
    double arm = .3;
    Vector3d orx;
    orx<<1,0,0;
    Vector3d ory;
    ory<<0,1,0;
    Drone* d = new Drone(mass,max_force,pos,vel,ang,r_torque,orx,ory,arm);
    return d;
}

Drone* make_easy_drone() {
    Drone* d = new Drone(1,1,1,1);
    return d;
}

int main() {
    Drone* d = make_easy_drone();
    d->setspeeds(1,0,1,0);
    for (int i =0;i<100;++i) {
        d->update(.01,d->get_accelerations());}
    std::cout<<d->getpx()<<std::endl;
    std::cout<<d->getpy()<<std::endl;
    std::cout<<d->getpz()<<std::endl;
    std::cout<<std::endl;
    std::cout<<d->getorientation()<<std::endl;
    return 0;
}

