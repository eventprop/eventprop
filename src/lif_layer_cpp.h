#include <Eigen/Dense>
#include <limits>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

double inf = std::numeric_limits<double>::infinity();
double eps = std::numeric_limits<double>::epsilon();

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

py::tuple compute_spikes(Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<Eigen::ArrayXd const> times, Eigen::Ref<Eigen::ArrayXi const> sources, double v_th, double tau_mem, double tau_syn);
void backward(Eigen::Ref<Eigen::ArrayXd const> input_times, Eigen::Ref<Eigen::ArrayXi const> input_sources, Eigen::Ref<Eigen::ArrayXd const> post_times, Eigen::Ref<Eigen::ArrayXi const> post_sources, Eigen::Ref<Eigen::ArrayXd> input_errors, Eigen::Ref<Eigen::ArrayXd const> post_errors, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double v_th, double tau_mem, double tau_syn);