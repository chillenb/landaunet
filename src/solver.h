#include <array>
#include <assert.h>
#include <bitset>
#include <cmath>
#include <iostream>
#include <random>
#include <stdint.h>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using std::pow;

#define IDX(i, j, ny) ((i) * (ny) + (j))

namespace py = pybind11;
using std::array;
using std::bitset;
using std::vector;

using Eigen::Matrix3d;
using Eigen::Vector3d;

Vector3d bsolve(Vector3d &avec, Vector3d &bvec, double dt) {
  double b1 = bvec[0], b2 = bvec[1], b3 = bvec[2];
  double hdt = 0.5 * dt;
  Eigen::Matrix3d M;
  M << 1.0 + pow(b1, 2) * pow(hdt, 2) - (pow(b2, 2) + pow(b3, 2)) * pow(hdt, 2),
      2 * hdt * (b3 + b1 * b2 * hdt),
      2 * hdt * (-b2 + b1 * b3 * hdt),
      2 * hdt * (-b3 + b1 * b2 * hdt),
      1.0 - (pow(b1, 2) - pow(b2, 2) + pow(b3, 2)) * pow(hdt, 2),
      2 * hdt * (b1 + b2 * b3 * hdt),
      2 * hdt * (b2 + b1 * b3 * hdt),
      2 * hdt * (-b1 + b2 * b3 * hdt),
      1.0 - (pow(b1, 2) + pow(b2, 2) - pow(b3, 2)) * pow(hdt, 2);


  double denom = 1 + (pow(b1, 2) + pow(b2, 2) + pow(b3, 2)) * pow(hdt, 2);
  Eigen::Vector3d v = (M * avec) / denom;
  return {v[0], v[1], v[2]};
}

void redblack_step_2d(py::array_t<double> data, double h) {
  auto r = data.mutable_unchecked<3>();
  py::ssize_t nx = r.shape(0);
  py::ssize_t ny = r.shape(1);
  if (r.shape(2) != 3) {
    throw std::runtime_error("Input array must be NxNx3");
  }
}