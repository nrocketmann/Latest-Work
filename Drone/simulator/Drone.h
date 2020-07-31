//
// Created by Nameer Hirschkind on 5/4/20.
//

#ifndef DRONESIM_DRONE_H
#define DRONESIM_DRONE_H
#include "eigen-3.3.7/Eigen/Dense"
#include "eigen-3.3.7/Eigen/Geometry"
#include <math.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;


class Drone {

private:
    double mass;
    double max_motor_force;
    Vector3d position; //CHANGES
    Vector3d velocity; //CHANGES
    Vector3d angular_velocity; //CHANGES
    double rotor_torque;
    VectorXd rotor_speeds; //SPECIFIED FROM OUTSIDE
    MatrixXd Icm;
    MatrixXd bigmat;
    MatrixXd bigmat_inv; //so we can solve linear equations easily
    //note this inverse should always exist, since it's a diagonal matrix
    MatrixXd orientation; // CHANGES
    double arm_length;
    MatrixXd thrustmat;

public:

    //constructor (no vectors to make python binding easier)
    Drone(double m, double mrf, double posx, double posy, double posz, double velx,
            double vely, double velz, double angx, double angy, double angz, double r_torque,
            double orientationx, double orientationy, double orientationz, double forwardx,
            double forwardy, double forwardz,double arm);

    //constructor with vectors
    Drone(double m, double mrf, Vector3d pos, Vector3d vel, Vector3d ang, double r_torque,
          Vector3d x_orientation, Vector3d y_orientation, double arm);

    //stationary constructor
    Drone(double m, double mrf, double r_torque, double arm);

    //outputs (acm, a_angular) in one 6-length vector
    VectorXd get_accelerations();

    //step the simulation forward
    void update(double dt, const VectorXd &accelerations);

    //getters and setters
    //position
    double getpx() {return position[0];}
    double getpy() {return position[1];}
    double getpz() {return position[2];}
    const Vector3d& getp() {return position;}

    //velocity
    double getvx() {return velocity[0];}
    double getvy() {return velocity[1];}
    double getvz() {return velocity[2];}
    const Vector3d& getv() {return velocity;}

    //angular
    double getwx() {return angular_velocity[0];}
    double getwy() {return angular_velocity[1];}
    double getwz() {return angular_velocity[2];}
    const Vector3d& getw() {return angular_velocity;}

    //orientation
    const MatrixXd& getorientation() {return orientation;}
    //rotor speeds
    void setspeeds(double speed1,double speed2,double speed3,double speed4) {
        rotor_speeds[0] = speed1;
        rotor_speeds[1] = speed2;
        rotor_speeds[2] = speed3;
        rotor_speeds[3] = speed4;
    }

    //SETTERS

    void setspeeds_vector(VectorXd speeds) {
        rotor_speeds = speeds;
    }

    void setpos(Vector3d pos) {position = pos;}
    void setvel(Vector3d vel) {velocity = vel;}
    void setangular(Vector3d ang) {angular_velocity = ang;}
    void setor(Vector3d orx, Vector3d ory) {
        Vector3d orz = orx.cross(ory);
        orientation.col(0) = orx;
        orientation.col(1) = ory;
        orientation.col(2) = orz;
    }

    void perform_update(double dt);
    void update_n_times(double dt, int n) {
        for (int i = 0;i<n;++i) {perform_update(dt);}
    }

    //we need to get reward in c++ for speed
    //this will just be 1/loss^2 or the sum of the squared vector norms
    double get_loss();

    VectorXd get_vector();

    VectorXd speeds_update_return(VectorXd speeds, double dt, int n) {
        setspeeds_vector(speeds);
        update_n_times(dt,n);
        return get_vector();
    }

    double get_reward();
};





#endif //DRONESIM_DRONE_H
