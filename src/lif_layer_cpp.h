#include <Eigen/Dense>
#include <limits>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

double inf = std::numeric_limits<double>::infinity();
double eps = std::numeric_limits<double>::epsilon();

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Spikes {
    Eigen::ArrayXd times;
    Eigen::ArrayXi sources;
    Eigen::ArrayXd errors;
    int n_spikes;

    explicit Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources, Eigen::ArrayXd errors) : times(times), sources(sources), errors(errors), n_spikes(times.size()) { }
    explicit Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources) : times(times), sources(sources), errors(Eigen::ArrayXd::Zero(times.size())), n_spikes(times.size()) { }
    Spikes() {}
    void set_error(int spike_idx, double error) { errors(spike_idx) = error; }
};


std::pair<Eigen::ArrayXd, Eigen::ArrayXi> compute_spikes(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double v_th, double tau_mem, double tau_syn);
py::list compute_spikes_batch(Eigen::Ref<RowMatrixXd const> w, py::list batch, double v_th, double tau_mem, double tau_syn);
void backward(Spikes const& input_spikes, Spikes const& post_spikes, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double v_th, double tau_mem, double tau_syn);
void backward_batch(py::list input_batch, py::list post_batch, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double v_th, double tau_mem, double tau_syn);