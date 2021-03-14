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
  std::vector<double> first_spike_times;
  std::vector<int> first_spike_idxs;
  int n_spikes;

  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources,
                  Eigen::ArrayXd errors, std::vector<double> first_spike_times, std::vector<int> first_spike_idxs)
      : times(times), sources(sources), errors(errors), first_spike_times(first_spike_times), first_spike_idxs(first_spike_idxs), n_spikes(times.size()) {
  }
  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources)
      : times(times), sources(sources),
        errors(Eigen::ArrayXd::Zero(times.size())), first_spike_times(), first_spike_idxs(), n_spikes(times.size())  {}
  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources, std::vector<double> first_spike_times, std::vector<int> first_spike_idxs)
      : times(times), sources(sources),
        errors(Eigen::ArrayXd::Zero(times.size())), first_spike_times(first_spike_times), first_spike_idxs(first_spike_idxs), n_spikes(times.size()){}
  Spikes() {}
  void set_error(int spike_idx, double error) { errors(spike_idx) = error; }
  void set_time(int spike_idx, double time) { times(spike_idx) = time; }
};

PYBIND11_MAKE_OPAQUE(std::vector<Spikes>);

Spikes
compute_spikes(Eigen::Ref<RowMatrixXd const> w, Spikes const &spikes,
               double v_th, double tau_mem, double tau_syn);
std::vector<Spikes>
compute_spikes_batch(Eigen::Ref<RowMatrixXd const> w,
                     std::vector<Spikes> const &batch, double v_th,
                     double tau_mem, double tau_syn);
void backward(Spikes const &input_spikes, Spikes const &post_spikes,
              Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient,
              double v_th, double tau_mem, double tau_syn);
void backward_batch(std::vector<Spikes> &input_batch,
                    std::vector<Spikes> const &post_batch,
                    Eigen::Ref<RowMatrixXd const> w,
                    Eigen::Ref<RowMatrixXd> gradient, double v_th,
                    double tau_mem, double tau_syn);

std::pair<RowMatrixXd, RowMatrixXi>
find_first_spikes(std::vector<Spikes> const &input_batch, int n);