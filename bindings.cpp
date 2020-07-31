#include "Drone.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

namespace py = pybind11;


PYBIND11_MODULE(drone, m) {
py::class_<Drone>(m, "Drone")
.def(py::init<double,double,MatrixXd,MatrixXd,MatrixXd,double,MatrixXd,MatrixXd,double>())
.def(py::init<double,double,double,double>())
.def("getPosition",&Drone::getp)
.def("getVelocity",&Drone::getv)
.def("getAngular",&Drone::getw)
.def("setSpeed",&Drone::setspeeds_vector)
.def("getOrientation", &Drone::getorientation)
.def("update", &Drone::perform_update)
.def("setPosition",&Drone::setpos)
.def("setVelocity",&Drone::setvel)
.def("setAngular",&Drone::setangular)
.def("setOrientation",&Drone::setor)
.def("getLoss",&Drone::get_loss)
.def("getVector",&Drone::get_vector)
.def("updateN",&Drone::update_n_times)
.def("fullIteration",&Drone::speeds_update_return)
.def("getReward",&Drone::get_reward);
}



