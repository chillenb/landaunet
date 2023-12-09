#include <pybind11/pybind11.h>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#include "solver.h"
#include <stdint.h>


namespace py = pybind11;
using std::vector;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        LL solver in C++
        -----------------------

        .. currentmodule:: landaunet

        .. autosummary::
           :toctree: _generate

    )pbdoc";
    m.def("bsolve", &bsolve, R"pbdoc(
        red-black update core equation
    )pbdoc");
    m.def("redblack_step_2d", &redblack_step_2d, R"pbdoc(
        Solve the LL equation for a single timestep
    )pbdoc");


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
