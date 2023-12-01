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
        Ising model in C++
        -----------------------

        .. currentmodule:: ising

        .. autosummary::
           :toctree: _generate

           get_spins
           random_mc_meanstd
           energy
    )pbdoc";
    py::class_<Model>(m, "Model")
        .def(py::init<uint64_t, Real, Real, vector<int>>())
        .def("get_spins", &Model::get_spins)
        .def("random_mc_meanstd", &Model::random_mc_meanstd)
        .def_readwrite("energy", &Model::energy);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
