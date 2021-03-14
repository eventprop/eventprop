#include <Eigen/Dense>
#include <limits>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

double inf = std::numeric_limits<double>::infinity();
double eps = std::numeric_limits<double>::epsilon();

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>;

struct Spikes {
  Eigen::ArrayXd times;
  Eigen::ArrayXi sources;
  Eigen::ArrayXd errors;
  Eigen::ArrayXd currents;
  std::vector<double> first_spike_times;
  std::vector<int> first_spike_idxs;
  int n_spikes;
  double dead_fraction;

  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources)
      : times(times), sources(sources),
        errors(Eigen::ArrayXd::Zero(times.size())), first_spike_times(), first_spike_idxs(), n_spikes(times.size())  {}
  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources, std::vector<double> first_spike_times, std::vector<int> first_spike_idxs)
      : times(times), sources(sources),
        errors(Eigen::ArrayXd::Zero(times.size())), first_spike_times(first_spike_times), first_spike_idxs(first_spike_idxs), n_spikes(times.size()){}
  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources, Eigen::ArrayXd currents, std::vector<double> first_spike_times, std::vector<int> first_spike_idxs, double dead_fraction)
      : times(times), sources(sources),
        errors(Eigen::ArrayXd::Zero(times.size())), currents(currents), first_spike_times(first_spike_times), first_spike_idxs(first_spike_idxs), n_spikes(times.size()), dead_fraction(dead_fraction) {}
  Spikes() {}
  void set_error(int spike_idx, double error) { errors(spike_idx) = error; }
  void set_time(int spike_idx, double time) { times(spike_idx) = time; }
};

struct Maxima {
  Eigen::ArrayXd times;
  Eigen::ArrayXd values;
  Eigen::ArrayXd errors;

  Maxima(Eigen::ArrayXd times, Eigen::ArrayXd values, Eigen::ArrayXd errors) : times(times), values(values), errors(errors) {}
  Maxima(Eigen::ArrayXd times, Eigen::ArrayXd values) : times(times), values(values), errors(Eigen::ArrayXd::Zero(times.size())) {}
  Maxima() {}
  void set_error(int nrn_idx, double error) {errors(nrn_idx) = error; }
};

PYBIND11_MAKE_OPAQUE(Spikes);
PYBIND11_MAKE_OPAQUE(Maxima);
PYBIND11_MAKE_OPAQUE(std::vector<Spikes>);
PYBIND11_MAKE_OPAQUE(std::vector<Maxima>);

std::pair<RowMatrixXd, RowMatrixXd> compute_sums(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double tau_mem, double tau_syn);
Spikes
compute_spikes(Eigen::Ref<RowMatrixXd const> w, Spikes const &spikes,
               double v_th, double tau_mem, double tau_syn);
std::pair<std::vector<Spikes>, double>
compute_spikes_batch(Eigen::Ref<RowMatrixXd const> w,
                     std::vector<Spikes> const &batch, double v_th,
                     double tau_mem, double tau_syn);
void backward(Spikes const &input_spikes, Spikes const &post_spikes,
              Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient,
              double v_th, double tau_mem, double tau_syn);
void backward_spikes_batch(std::vector<Spikes> &input_batch,
                    std::vector<Spikes> const &post_batch,
                    Eigen::Ref<RowMatrixXd const> w,
                    Eigen::Ref<RowMatrixXd> gradient, double v_th,
                    double tau_mem, double tau_syn);
Maxima compute_maxima(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double tau_mem, double tau_syn);
void backward(Spikes & input_spikes, Maxima const& maxima, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double tau_mem, double tau_syn);
std::vector<Maxima>
compute_maxima_batch(Eigen::Ref<RowMatrixXd const> w, std::vector<Spikes> const& batch, double tau_mem, double tau_syn);
void
backward_maxima_batch(std::vector<Spikes> &input_batch, std::vector<Maxima> const& maxima, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double tau_mem, double tau_syn);