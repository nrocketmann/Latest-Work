//
// Created by Nameer Hirschkind on 5/4/20.
//

#include "Drone.h"
#include <iostream>

//REQUIRES:
// 1) orientation vectors are orthogonal unit vectors (orientationX and forwardX are x_hat and y_hat vectors)
//   Y axis
//    ^
// \  |  /
//  \ | /
//   --- -----> X axis
//  /   \
// /     \


// 2) nonzero mass, max motor force, arm length, and rotor torque
// 3) velocity, angular velocity, and position can be anything
Drone::Drone(double m, double mrf, double posx, double posy, double posz, double velx,
      double vely, double velz, double angx, double angy, double angz, double r_torque,
      double orientationx, double orientationy, double orientationz, double forwardx,
      double forwardy, double forwardz,double arm): mass(m), max_motor_force(mrf),
      position(Vector3d()), velocity(Vector3d()), angular_velocity(Vector3d()),
      rotor_speeds(VectorXd(4)), Icm(MatrixXd(3,3)), bigmat(MatrixXd(6,6)),
      bigmat_inv(MatrixXd(6,6)), orientation(MatrixXd(3,3)), arm_length(arm),thrustmat(MatrixXd(3,4)){
    mass = m;
    max_motor_force = mrf;
    position<<posx,posy,posz;
    velocity<<velx,vely,velz;
    angular_velocity<<angx,angy,angz;
    rotor_torque = r_torque;
    rotor_speeds<<0,0,0,0;
    arm_length = arm;

    //set orientation
    Vector3d orientation_one(orientationx,orientationy,orientationz);
    Vector3d orientation_two(forwardx,forwardy,forwardz);
    Vector3d orientation_three = orientation_one.cross(orientation_two);
    orientation.col(0) = orientation_one;
    orientation.col(1) = orientation_two;
    orientation.col(2) = orientation_three;

    //complicated moment of inertia stuff...
    //model the copter as a cylinder in the center, and 4 point masses
    //https://en.wikipedia.org/wiki/List_of_moments_of_inertia#List_of_3D_inertia_tensors
    //all units meters
    MatrixXd Icenter(3,3);
    double idiag1 = .25*m*(3.0*(arm*.25)*(arm*.25) + .05*.05);
    double idiag2 = m*(3.0*(arm*.25)*(arm*.25) + .05*.05)/24.0;
    double idiag3 = .5*m/2.0*(arm*.25)*(arm*.25);
    Icenter<<idiag1,0,0,
    0, idiag2,0,
    0,0,idiag3;

    double rotor_eff_mass = mass/8; //double it to make moment of inertia larger (air resistance)
    MatrixXd Irotors(3,3); //modeled as point masses
    double ixx = rotor_eff_mass * arm * arm * .5 * 4;
    double iyy = ixx;
    double izz = arm * arm * rotor_eff_mass * 4;
    Irotors<<ixx,0,0,
    0,iyy,0,
    0,0,izz;

    Icm = Icenter + Irotors;

    bigmat = MatrixXd::Zero(6,6);
    bigmat.topLeftCorner(3,3) = mass*MatrixXd::Identity(3,3);
    bigmat.bottomRightCorner(3,3) = Icm;
    bigmat_inv = bigmat.inverse();
    thrustmat<< 0,0,0,0,
            0,0,0,0,
            1,1,1,1;
}

Drone::Drone(double m, double mrf, Vector3d pos, Vector3d vel, Vector3d ang, double r_torque, Vector3d x_orientation,
             Vector3d y_orientation, double arm)  {
    mass = m;
    max_motor_force = mrf;
    arm_length = arm;
    rotor_torque = r_torque;
    rotor_speeds = VectorXd(4);
    rotor_speeds<<0,0,0,0;

    position = pos;
    velocity = vel;
    angular_velocity = ang;

    //set orientation
    orientation = MatrixXd(3,3);
    Vector3d z_orientation = x_orientation.cross(y_orientation);
    orientation.col(0) = x_orientation;
    orientation.col(1) = y_orientation;
    orientation.col(2) = z_orientation;

    //complicated moment of inertia stuff...
    //model the copter as a cylinder in the center, and 4 point masses
    //https://en.wikipedia.org/wiki/List_of_moments_of_inertia#List_of_3D_inertia_tensors
    //all units meters
    MatrixXd Icenter(3,3);
    double idiag1 = .25*m*(3.0*(arm*.25)*(arm*.25) + .05*.05);
    double idiag2 = m*(3.0*(arm*.25)*(arm*.25) + .05*.05)/24.0;
    double idiag3 = .5*m/2.0*(arm*.25)*(arm*.25);
    Icenter<<idiag1,0,0,
            0, idiag2,0,
            0,0,idiag3;


    double rotor_eff_mass = mass/8;
    MatrixXd Irotors(3,3); //modeled as point masses
    double ixx = rotor_eff_mass * arm * arm * .5 * 4;
    double iyy = ixx;
    double izz = arm * arm * rotor_eff_mass * 4;
    Irotors<<ixx,0,0,
    0,iyy,0,
    0,0,izz;
    MatrixXd Icm(3,3);
    Icm = Icenter + Irotors;

    bigmat = MatrixXd::Zero(6,6);
    bigmat.topLeftCorner(3,3) = mass*MatrixXd::Identity(3,3);
    bigmat.bottomRightCorner(3,3) = Icm;
    bigmat_inv = bigmat.inverse();

    thrustmat = MatrixXd(3,4);
    thrustmat<< 0,0,0,0,
            0,0,0,0,
            1,1,1,1;
}

Drone::Drone(double m, double mrf, double r_torque, double arm) {
    mass = m;
    max_motor_force = mrf;
    rotor_torque = r_torque;
    arm_length = arm;
    rotor_speeds = VectorXd(4);
    rotor_speeds<<0,0,0,0;

    Vector3d pos(0,0,0);
    Vector3d vel(0,0,0);
    Vector3d ang(0,0,0);

    position = pos;
    velocity = vel;
    angular_velocity = ang;

    //set orientation
    orientation = MatrixXd(3,3);
    orientation<<1,0,0,
    0,1,0,
    0,0,1;

    //complicated moment of inertia stuff...
    //model the copter as a cylinder in the center, and 4 point masses
    //https://en.wikipedia.org/wiki/List_of_moments_of_inertia#List_of_3D_inertia_tensors
    //all units meters
    MatrixXd Icenter(3,3);
    double idiag1 = .25*m*(3.0*(arm*.25)*(arm*.25) + .05*.05);
    double idiag2 = m*(3.0*(arm*.25)*(arm*.25) + .05*.05)/24.0;
    double idiag3 = .5*m/2.0*(arm*.25)*(arm*.25);
    Icenter<<idiag1,0,0,
            0, idiag2,0,
            0,0,idiag3;
    double rotor_eff_mass = mass/4;
    MatrixXd Irotors(3,3); //modeled as point masses
    double ixx = rotor_eff_mass * arm * arm * .5 * 4;
    double iyy = ixx;
    double izz = arm * arm * rotor_eff_mass * 4;
    Irotors<<ixx,0,0,
    0,iyy,0,
    0,0,izz;
    Icm = MatrixXd(3,3);
    Icm = Icenter + Irotors;

    bigmat = MatrixXd::Zero(6,6);
    bigmat.topLeftCorner(3,3) = mass*MatrixXd::Identity(3,3);
    bigmat.bottomRightCorner(3,3) = Icm;
    bigmat_inv = bigmat.inverse();

    thrustmat = MatrixXd(3,4);
    thrustmat<< 0,0,0,0,
            0,0,0,0,
            1,1,1,1;
}


VectorXd Drone::get_accelerations() {
    //b vector
    VectorXd b(6);
    Vector3d angular_b = angular_velocity.cross(static_cast<Vector3d>(Icm*angular_velocity));
    //Vector3d angular_b(0,0,0);
    Vector3d zero_part = Vector3d::Zero();
    b.head(3) = zero_part;
    b.tail(3) = angular_b;

    //motor thrusts and gravity
    Vector3d Fa = thrustmat*rotor_speeds*max_motor_force;
    Vector3d normal_g;
    normal_g<<0,0,-mass*9.8;
    Vector3d Fg = orientation.inverse()*normal_g;

    Vector3d F = Fa+Fg;

    //torque
    //motor config:
    // 3 2
    //  X
    // 0 1
    //0 and 2 turn clockwise, 1 and 3 counter
    Vector3d T = Vector3d::Zero();
    for (int i = 0;i<4;++i) {
        Vector3d yaw_torque;

        if (i==0 || i==2) {
            yaw_torque<<0,0,rotor_torque*rotor_speeds[i];
        }
        else {
            yaw_torque<<0,0,-rotor_torque*rotor_speeds[i];
        }
        Vector3d r;
        double coord1 = (i==2 || i==1)?(sqrt(2)/2):(-sqrt(2)/2);
        double coord2 = (i<2)?(-sqrt(2)/2):(sqrt(2)/2);
        r<<coord1,coord2,0;
        Vector3d f;
        f<<0,0,rotor_speeds[i]*max_motor_force;
        Vector3d pitch_torque = r.cross(f);
        T+=pitch_torque+yaw_torque;
    }
    VectorXd all_forces(6);
    all_forces.head(3) = F;
    all_forces.tail(3) = T;

    //now we have all the forces and torque
    //time to solve for acceleration using bigmat inverse
    VectorXd y_eqn = all_forces-b;
    VectorXd all_accelerations = bigmat_inv*y_eqn;

    return all_accelerations;
}

void Drone::update(double dt, const VectorXd &accelerations) {
    Vector3d ac = accelerations.head(3);
    Vector3d ang = accelerations.tail(3);

    velocity+=orientation*ac*dt;
    position+=velocity*dt; //orientation already applied to get velocity
    angular_velocity+=orientation*ang*dt;

    Vector3d angle_turned = angular_velocity*dt;
    //make a rotation matrix out of this

    MatrixXd rotx(3,3);
    MatrixXd roty(3,3);
    MatrixXd rotz(3,3);
    rotx<<1,0,0,
    0,cos(angle_turned[0]),-sin(angle_turned[0]),
    0,sin(angle_turned[0]),cos(angle_turned[0]);
    roty<<cos(angle_turned[1]),0,-sin(angle_turned[1]),
    0,1,0,
    sin(angle_turned[1]),0,cos(angle_turned[1]);
    rotz<<cos(angle_turned[2]),-sin(angle_turned[2]),0,
    sin(angle_turned[2]),cos(angle_turned[2]),0,
    0,0,1;
    // rotx, roty, rotz are all in orientation-space. We want this transformation to be in euclidean space
    // rotx*roty*rotz*I = rotx*roty*rotz is our new orientation in orientation-space, so multiply by orientation
    // to get it in euclidean space

    orientation = rotx*roty*rotz*orientation;

}

void Drone::perform_update(double dt) {
    VectorXd ac = this->get_accelerations();
    this->update(dt,ac);
}

double Drone::get_loss() {
    double pos_loss = position.squaredNorm();
    double vel_loss = velocity.squaredNorm();
    double ang_loss = angular_velocity.squaredNorm();
    double or_loss = (orientation-MatrixXd::Identity(3,3)).squaredNorm();
    double loss_sum = pos_loss+vel_loss+ang_loss+or_loss;
    //double loss_sum = pos_loss + or_loss;
    return loss_sum;
}

VectorXd Drone::get_vector() {
    VectorXd out(18);
    out.head(3) = position;
    out.segment(3,3) = velocity;
    out.segment(6,3) = angular_velocity;
    out.segment(9,3) = orientation.col(0);
    out.segment(12,3) = orientation.col(1);
    out.segment(15,3) = orientation.col(2);
    return out;
}

double Drone::get_reward() {
    double loss = get_loss();
    if (loss>100) return 0;
    else return 1/(loss+1);
}