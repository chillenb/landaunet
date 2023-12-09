#include <array>
#include <assert.h>
#include <bitset>
#include <cmath>
#include <iostream>
#include <random>
#include <stdint.h>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using std::pow;

#define IDX(i, j, ny) ((i) * (ny) + (j))

namespace py = pybind11;
using std::array;
using std::bitset;
using std::vector;

using Eigen::Matrix3d;
using Eigen::Vector3d;

inline std::array<double,3> bsolve(double a1, double a2, double a3, double b1, double b2, double b3, double dt) {
  double hdt = 0.5 * dt;

  std::array<double,9> M = {1.0 + pow(b1, 2) * pow(hdt, 2) - (pow(b2, 2) + pow(b3, 2)) * pow(hdt, 2),
      2 * hdt * (b3 + b1 * b2 * hdt),
      2 * hdt * (-b2 + b1 * b3 * hdt),
      2 * hdt * (-b3 + b1 * b2 * hdt),
      1.0 - (pow(b1, 2) - pow(b2, 2) + pow(b3, 2)) * pow(hdt, 2),
      2 * hdt * (b1 + b2 * b3 * hdt),
      2 * hdt * (b2 + b1 * b3 * hdt),
      2 * hdt * (-b1 + b2 * b3 * hdt),
      1.0 - (pow(b1, 2) + pow(b2, 2) - pow(b3, 2)) * pow(hdt, 2)};
  
  std::array<double,3> m_times_a = {M[0] * a1 + M[1] * a2 + M[2] * a3,
      M[3] * a1 + M[4] * a2 + M[5] * a3,
      M[6] * a1 + M[7] * a2 + M[8] * a3};

  double denom_inv = 1.0/(1.0 + (pow(b1, 2) + pow(b2, 2) + pow(b3, 2)) * pow(hdt, 2));
  
  std::array<double, 3> result = {m_times_a[0] * denom_inv,
      m_times_a[1] * denom_inv,
      m_times_a[2] * denom_inv};
  return result;
}

void redblack_step_2d(py::array_t<double> data, double dt) {
  auto r = data.mutable_unchecked<3>();
  py::ssize_t nx = r.shape(1);
  py::ssize_t ny = r.shape(2);
  if (r.shape(0) != 3 || nx != ny) {
    throw std::runtime_error("Input array must be 3xNxN");
  }
  
  // interior Q update:
  for (py::ssize_t i = 1; i < nx - 1; i++) {
    for (py::ssize_t j = 1; j < ny - 1; j++) {
      if ((i + j) % 2 == 0) {

        std::array<double,3> Pavg = {0.0, 0.0, 0.0};
        Pavg[0] = 0.25 * (r(0, i-1, j) + r(0, i+1, j) + r(0, i, j-1) + r(0, i, j+1));
        Pavg[1] = 0.25 * (r(1, i-1, j) + r(1, i+1, j) + r(1, i, j-1) + r(1, i, j+1));
        Pavg[2] = 0.25 * (r(2, i-1, j) + r(2, i+1, j) + r(2, i, j-1) + r(2, i, j+1));

        std::array<double,3> result = bsolve(r(0, i, j), r(1, i, j), r(2, i, j), Pavg[0], Pavg[1], Pavg[2], dt);
        r(0, i, j) = result[0];
        r(1, i, j) = result[1];
        r(2, i, j) = result[2];
      }
    }
  }

  // fill halo for dirichlet BC
  for (py::ssize_t i = 1; i < nx - 1; i++) {
    r(0, i, 0) = 0.0;
    r(1, i, 0) = 0.0;
    r(2, i, 0) = 1.0;
    r(0, i, ny - 1) = 0.0;
    r(1, i, ny - 1) = 0.0;
    r(2, i, ny - 1) = 1.0;
  }
  for (py::ssize_t j = 1; j < ny - 1; j++) {
    r(0, 0, j) = 0.0;
    r(1, 0, j) = 0.0;
    r(2, 0, j) = 1.0;
    r(0, nx - 1, j) = 0.0;
    r(1, nx - 1, j) = 0.0;
    r(2, nx - 1, j) = 1.0;
  }

  // interior P update:

  for (py::ssize_t i = 1; i < nx - 1; i++) {
    for (py::ssize_t j = 1; j < ny - 1; j++) {
      if ((i + j) % 2 == 1) {

        std::array<double,3> Qavg = {0.0, 0.0, 0.0};
        Qavg[0] = 0.25 * (r(0, i-1, j) + r(0, i+1, j) + r(0, i, j-1) + r(0, i, j+1));
        Qavg[1] = 0.25 * (r(1, i-1, j) + r(1, i+1, j) + r(1, i, j-1) + r(1, i, j+1));
        Qavg[2] = 0.25 * (r(2, i-1, j) + r(2, i+1, j) + r(2, i, j-1) + r(2, i, j+1));

        std::array<double,3> result = bsolve(r(0, i, j), r(1, i, j), r(2, i, j), Qavg[0], Qavg[1], Qavg[2], dt);
        r(0, i, j) = result[0];
        r(1, i, j) = result[1];
        r(2, i, j) = result[2];
      }
    }
  }

  
  // fill halo for dirichlet BC
  for (py::ssize_t i = 1; i < nx - 1; i++) {
    r(0, i, 0) = 0.0;
    r(1, i, 0) = 0.0;
    r(2, i, 0) = 1.0;
    r(0, i, ny - 1) = 0.0;
    r(1, i, ny - 1) = 0.0;
    r(2, i, ny - 1) = 1.0;
  }
  for (py::ssize_t j = 1; j < ny - 1; j++) {
    r(0, 0, j) = 0.0;
    r(1, 0, j) = 0.0;
    r(2, 0, j) = 1.0;
    r(0, nx - 1, j) = 0.0;
    r(1, nx - 1, j) = 0.0;
    r(2, nx - 1, j) = 1.0;
  }

}