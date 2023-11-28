#include <array>
#include <assert.h>
#include <bitset>
#include <cmath>
#include <iostream>
#include <random>
#include <stdint.h>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define K 20

#define IDX(i, j) ((i) * K + (j))

inline int modK(int a) { return (a > K - 1) ? a - K : a; }

namespace py = pybind11;
using std::array;
using std::bitset;
using std::vector;

typedef double Real;

class Model {
public:
  bitset<K * K> spins;
  double J;
  double hmu;
  double energy;
  int twojpairs = 0;
  int64_t magnetization = 0;
  std::mt19937 rng;

  Model(uint64_t seed, double J, double hmu, vector<int> spins_input)
      : J(J), hmu(hmu) {
    twojpairs = 0;
    magnetization = 0;
    for (int i = 0; i < K * K; i++) {
      spins[i] = (spins_input[i] > 0) ? true : false;
    }

    for (int i = 0; i < K; i++) {
      for (int j = 0; j < K; j++) {
        int s = spins[IDX(i, j)];
        int sup = spins[IDX(modK(i + (K - 1)), j)];
        int sdown = spins[IDX(modK(i + 1), j)];
        int sleft = spins[IDX(i, modK(j + (K - 1)))];
        int sright = spins[IDX(i, modK(j + 1))];
        int sumspins = (sup + sdown + sleft + sright) * 2 - 4;
        energy += -((0.5 * J) * sumspins + hmu) * (2 * s - 1);
        twojpairs += sumspins * (2 * s - 1);
        magnetization += (2 * s - 1);
      }
    }
    rng = std::mt19937(seed);
  }

  vector<int> get_spins() {
    vector<int> spins_output(K * K);
    for (int i = 0; i < K * K; i++) {
      spins_output[i] = this->spins[i] ? 1 : -1;
    }
    return spins_output;
  }

  std::tuple<Real, Real, Real> random_mc_meanstd(int nsamp, Real beta,
                                                 int samp_freq) {
    std::uniform_real_distribution<float> dist(0, 1);
    std::uniform_int_distribution<short> intdist(0, K - 1);

    std::array<double, 5> expdes = {exp(-2 * beta * (J * -4)),
                                    exp(-2 * beta * (J * -2)),
                                    exp(-2 * beta * (J * 0)),
                                    exp(-2 * beta * (J * 2)),
                                    exp(-2 * beta * (J * 4))};
    std::array<double, 2> expdes2 = {exp(-2 * beta * (hmu)),
                                     exp(2 * beta * (hmu))};

    int next_samp = 0;
    Real sumsq = 0.0;
    int64_t sum_2jpairs = 0;
    int64_t sum_mag = 0;
    for (int k = 0; k < nsamp; k++) {
      if (k == next_samp) {
        sum_2jpairs += twojpairs;
        sum_mag += abs(magnetization);
        Real samp = -0.5 * J * twojpairs - hmu * magnetization;
        sumsq += samp * samp;
        next_samp += samp_freq;
      }
      int i = intdist(rng);
      int j = intdist(rng);
      int s = spins[IDX(i, j)];
      int sup = spins[IDX(modK(i + (K - 1)), j)];
      int sdown = spins[IDX(modK(i + 1), j)];
      int sleft = spins[IDX(i, modK(j + (K - 1)))];
      int sright = spins[IDX(i, modK(j + 1))];
      int sumspins = (sup + sdown + sleft + sright) * 2 - 4;
      int sumspins_idx = (sup + sdown + sleft + sright - 2) * (2*s-1) + 2;
      double dE = 2 * (2 * s - 1) * (J * sumspins + hmu);

      
      #if 0
      if(abs(exp(-beta*dE) - expdes[sumspins_idx] * expdes2[2*s+1]) > 1e-6) {
        std::cout << "sumspins_idx is " << sumspins_idx << std::endl;
        std::cout << "sumspins*(2s+1) is " << sumspins*(2*s+1) << std::endl;
        std::cout << "exp(-beta*dE) = " << exp(-beta*dE) << std::endl;
        std::cout << "expdes[sumspins_idx] * expdes2[2] = " << expdes[sumspins_idx] * expdes2[2*s+1] << std::endl;
        throw std::runtime_error("Bad value");
      }
      #endif
      
      if (dE < 0) {
        spins[IDX(i, j)] = !s;
        twojpairs -= 4 * (2 * s - 1) * sumspins;
        magnetization -= 2 * (2 * s - 1);
        this->energy += dE;
      } else {
        double p = expdes[sumspins_idx] * expdes2[2*s+1];
        if (dist(rng) < p) {
          spins[IDX(i, j)] = !s;
          twojpairs -= 4 * (2 * s - 1) * sumspins;
          magnetization -= 2 * (2 * s - 1);
          this->energy += dE;
        }
      }
      #if 0
      if(magnetization > K*K || magnetization < -K*K) {
        std::cout << "magnetization is " << magnetization << std::endl;
        std::cout << "dE is " << dE << std::endl;
        std::cout << "sumspins is " << sumspins << std::endl;
        std::cout << "going from " << 2*s-1 << " to " << 2*(!s)-1 << std::endl;
        int real_mag = 0;
        for(int i = 0; i < K; i++) {
          for(int j = 0; j < K; j++) {
            real_mag += 2*(int)spins[IDX(i,j)]-1;
          }
        }
        std::cout << "real_mag is " << real_mag << std::endl;
        throw std::runtime_error("Bad value");
      }
      #endif
    }
    int64_t n = nsamp / samp_freq;
    Real mean = (-0.5 * J * sum_2jpairs - hmu * sum_mag) / ((double)n);
    Real std = sqrt(sumsq / ((double)(n - 1)) -
                    mean * mean * ((double)n) / ((double)(n - 1)));
    return std::make_tuple(mean, std, (double) sum_mag / ((double)(n*K*K)));
  }
};
