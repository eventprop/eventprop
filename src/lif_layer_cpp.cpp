#include "lif_layer_cpp.h"

#include <algorithm>
#include <boost/format.hpp>
#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

Eigen::VectorXd LIF::extract_times(SpikeVector const &spikes) const {
  Eigen::VectorXd result(spikes.size());
  int i = 0;
  std::for_each(spikes.begin(), spikes.end(),
                [&](Spike const &spike) { result[i++] = spike.time; });
  return result;
}

Eigen::VectorXi LIF::extract_sources(SpikeVector const &spikes) const {
  Eigen::VectorXi result(spikes.size());
  int i = 0;
  std::for_each(spikes.begin(), spikes.end(),
                [&](Spike const &spike) { result[i++] = spike.source_neuron; });
  return result;
}

void LIF::get_spikes() {
  for (int i = 0; i < n; i++) {
    get_spikes_for_neuron(i);
  }
}

void LIF::get_spikes_for_neuron(int target_nrn_idx) {
  if (not input_initialized) {
    throw std::runtime_error("Set input spikes first!");
  }
  bool finished = false;
  int processed_up_to = 0;
  double tmax, vmax, t_before, t_pre, t_post;
  post_spikes.at(target_nrn_idx).clear();
  while (not finished) {
    for (int i = processed_up_to; i < n_spikes + 1; i++) {
      if (i == n_spikes) {
        t_pre = inf;
      } else {
        t_pre = input_times[i];
      }
      tmax = get_tmax(i, target_nrn_idx);
      if (tmax != inf) {
        vmax = v(tmax, target_nrn_idx);
        if (vmax > v_th + eps) {
          t_before = 0;
          if (i > 0) {
            t_before = input_times[i - 1];
            for (auto const &spike : post_spikes.at(target_nrn_idx)) {
              if (spike.time > t_pre)
                break;
              if (t_before < spike.time) {
                t_before = spike.time;
              }
            }
          }
          t_post = bracket_spike(t_before, tmax, target_nrn_idx);
          post_spikes.at(target_nrn_idx)
              .push_back(Spike(target_nrn_idx, t_post, layer_id, 0));
          processed_up_to = i;
          break;
        }
      }
      if (v(t_pre, target_nrn_idx) > v_th + eps) {
        t_before = input_times[i - 1];
        tmax = t_pre;
        t_post = bracket_spike(t_before, tmax, target_nrn_idx);
        post_spikes.at(target_nrn_idx)
            .push_back(Spike(target_nrn_idx, t_post, layer_id, 0));
        processed_up_to = i;
        break;
      }
      if (i == n_spikes) {
        finished = true;
      }
    }
  }
  ran_forward.at(target_nrn_idx) = true;
}

void LIF::get_errors() {
  for (int i = 0; i < n; i++) {
    get_errors_for_neuron(i);
  }
}

void LIF::get_errors_for_neuron(int target_nrn_idx) {
  SpikeRefVector input(input_spikes.begin(), input_spikes.end());
  SpikeRefVector output(post_spikes.at(target_nrn_idx).begin(),
                        post_spikes.at(target_nrn_idx).end());
  SpikeRefVector sorted_spikes;
  // sort and merge
  std::merge(input.begin(), input.end(), output.begin(), output.end(),
             std::back_inserter(sorted_spikes),
             [](SpikeRef const &a, SpikeRef const &b) {
               return a.get().time < b.get().time;
             });
  double const largest_time = sorted_spikes.back().get().time;
  double lambda_v = 0;
  double previous_t = -inf;
  auto is_pre_spike = [&](SpikeRef spike) {
    return spike.get().source_layer != layer_id;
  };
  auto is_my_post_spike = [&](SpikeRef spike) {
    return (spike.get().source_layer == layer_id and
            spike.get().source_neuron == target_nrn_idx);
  };
  lambda_i_jumps.at(target_nrn_idx).clear();
  for (auto spike = sorted_spikes.rbegin(); spike != sorted_spikes.rend();
       ++spike) {
    auto const t_spike = spike->get().time;
    double t_bwd = largest_time - t_spike;
    if (is_pre_spike(*spike)) {
      lambda_v = std::exp(-(t_bwd - previous_t) / tau_mem) * lambda_v;
      double lambda_i = 0;
      for (auto const &jump : lambda_i_jumps.at(target_nrn_idx)) {
        lambda_i += jump.value * k_bwd(t_bwd - jump.time);
      }
      gradient(spike->get().source_neuron, target_nrn_idx) +=
          -tau_syn * lambda_i;
      if (spike->get().source_layer != 0) {
        spike->get().error += w(spike->get().source_neuron, target_nrn_idx) *
                              (lambda_v - lambda_i);
      }
    } else if (is_my_post_spike(*spike)) {
      double const current = i(t_spike, target_nrn_idx);
      lambda_v = std::exp(-(t_bwd - previous_t) / tau_mem) * lambda_v;
      double const jump_value =
          1 / (current - v_th) * (v_th * lambda_v + spike->get().error);
      lambda_i_jumps.at(target_nrn_idx)
          .push_back(LambdaJump{jump_value, t_bwd});
      lambda_v = current / (current - v_th) * lambda_v +
                 1 / (current - v_th) * spike->get().error;
    }
    previous_t = t_bwd;
  }
  ran_backward.at(target_nrn_idx) = true;
}

double LIF::bracket_spike(double a, double b,
                                int target_nrn_idx) const {
  boost::math::tools::eps_tolerance<double> t(
      std::numeric_limits<double>::digits - 1);
  boost::uintmax_t max_iter = boost::math::policies::get_max_root_iterations<
      boost::math::policies::policy<>>();
  std::pair<double, double> result = boost::math::tools::toms748_solve(
      [&](double t) { return v_delta(t, target_nrn_idx); }, a, b, t, max_iter);
  return (result.first + result.second) / 2;
}

double LIF::get_tmax(int input_spike_idx,
                           int target_nrn_idx) const {
  double t_input;
  if (input_spike_idx == 0) {
    return 0;
  } else if (input_spike_idx == n_spikes) {
    t_input = inf;
  } else {
    t_input = input_times[input_spike_idx];
  }
  double sum0_elem = sum0(input_spike_idx, target_nrn_idx);
  double sum1_elem = sum1(input_spike_idx, target_nrn_idx);
  double t_before = input_times[input_spike_idx - 1];
  for (auto &spike : post_spikes.at(target_nrn_idx)) {
    if (t_input <= spike.time)
      break;
    sum1_elem += -v_th * std::exp(spike.time / tau_mem);
  }
  double tmax =
      tmax_prefactor * (tmax_summand + std::log(sum1_elem / sum0_elem));
  if (t_before <= tmax && tmax <= t_input) {
    return tmax;
  }
  return inf;
}

double LIF::v(double t, int target_nrn_idx) const {
  assert(target_nrn_idx < n);
  double v = 0;
  for (auto const &spike : input_spikes) {
    if (spike.time > t)
      break;
    v += w(spike.source_neuron, target_nrn_idx) * k(t - spike.time);
  }
  for (auto const &spike : post_spikes.at(target_nrn_idx)) {
    if (spike.time > t)
      break;
    v += -v_th * std::exp(-(t - spike.time) / tau_mem);
  }
  return v;
}

double LIF::i(double t, int target_nrn_idx) const {
  assert(target_nrn_idx < n);
  double i = 0;
  for (auto const &spike : input_spikes) {
    if (spike.time > t)
      break;
    i += w(spike.source_neuron, target_nrn_idx) *
         std::exp(-(t - spike.time) / tau_syn);
  }
  return i;
}

double LIF::lambda_v(double t, int target_nrn_idx) const {
  assert(target_nrn_idx < n);
  SpikeVector input(input_spikes.begin(), input_spikes.end());
  SpikeVector output(post_spikes.at(target_nrn_idx).begin(),
                     post_spikes.at(target_nrn_idx).end());
  SpikeRefVector sorted_spikes;
  // sort and merge
  std::merge(input.begin(), input.end(), output.begin(), output.end(),
             std::back_inserter(sorted_spikes),
             [](SpikeRef const &a, SpikeRef const &b) {
               return a.get().time < b.get().time;
             });
  double const largest_time = sorted_spikes.back().get().time;
  double lambda_v = 0;
  double previous_t = -inf;
  double const t_bwd_target = largest_time - t;
  auto is_pre_spike = [&](SpikeRef spike) {
    return spike.get().source_layer != layer_id;
  };
  auto is_my_post_spike = [&](SpikeRef spike) {
    return (spike.get().source_layer == layer_id and
            spike.get().source_neuron == target_nrn_idx);
  };
  if (t >= largest_time) {
    return 0;
  }
  for (auto spike = sorted_spikes.rbegin(); spike != sorted_spikes.rend();
       ++spike) {
    auto const t_spike = spike->get().time;
    double t_bwd = largest_time - t_spike;
    if (t_spike < t) {
      return lambda_v * std::exp(-(t_bwd_target - previous_t) / tau_mem);
    }
    if (is_pre_spike(*spike)) {
      lambda_v = std::exp(-(t_bwd - previous_t) / tau_mem) * lambda_v;
    } else if (is_my_post_spike(*spike)) {
      double const current = i(t_spike, target_nrn_idx);
      lambda_v = std::exp(-(t_bwd - previous_t) / tau_mem) * lambda_v;
      lambda_v = current / (current - v_th) * lambda_v +
                 1 / (current - v_th) * spike->get().error;
    }
    previous_t = t_bwd;
  }
  return lambda_v * std::exp(-(t_bwd_target - previous_t) / tau_mem);
}

double LIF::lambda_i(double t, int target_nrn_idx) const {
  assert(target_nrn_idx < n);
  SpikeVector input(input_spikes.begin(), input_spikes.end());
  SpikeVector output(post_spikes.at(target_nrn_idx).begin(),
                     post_spikes.at(target_nrn_idx).end());
  SpikeRefVector sorted_spikes;
  // sort and merge
  std::merge(input.begin(), input.end(), output.begin(), output.end(),
             std::back_inserter(sorted_spikes),
             [](SpikeRef const &a, SpikeRef const &b) {
               return a.get().time < b.get().time;
             });
  double const largest_time = sorted_spikes.back().get().time;
  double const t_bwd_target = largest_time - t;
  double lambda_i = 0;
  if (t >= largest_time) {
    return 0;
  }
  for (auto const &jump : lambda_i_jumps.at(target_nrn_idx)) {
    if (jump.time > t_bwd_target) {
      return lambda_i;
    }
    lambda_i += jump.value * k_bwd(t_bwd_target - jump.time);
  }
  return lambda_i;
}

std::vector<double> LIF::get_lambda_i_trace(int target_nrn_idx,
                                            double t_max,
                                            double dt = 1e-4) const {
  if (not ran_backward.at(target_nrn_idx)) {
    throw std::runtime_error("Run backward first!");
  }
  size_t size = std::floor(t_max / dt);
  std::vector<double> trace(size);
  double t = 0;
  for (auto &value : trace) {
    value = lambda_i(t, target_nrn_idx);
    t += dt;
  }
  return trace;
}

std::vector<double> LIF::get_voltage_trace(int target_nrn_idx,
                                           double t_max,
                                           double dt = 1e-4) const {
  if (not ran_forward.at(target_nrn_idx)) {
    throw std::runtime_error("Run forward first!");
  }
  size_t size = std::floor(t_max / dt);
  std::vector<double> trace(size);
  double t = 0;
  for (auto &voltage : trace) {
    voltage = v(t, target_nrn_idx);
    t += dt;
  }
  return trace;
}

double inline LIF::v_delta(double t,
                                 int target_nrn_idx) const {
  return v(t, target_nrn_idx) - v_th;
}

double inline LIF::k(double t) const {
  return k_prefactor * (std::exp(-t / tau_mem) - std::exp(-t / tau_syn));
}

double inline LIF::k_bwd(double t) const {
  return k_bwd_prefactor * k(t);
}

void LIF::set_post_spikes(SpikeVector output) {
  std::for_each(post_spikes.begin(), post_spikes.end(),
                [](SpikeVector &spikes) { spikes.clear(); });
  std::for_each(lambda_i_jumps.begin(), lambda_i_jumps.end(),
                [](LambdaJumpVector &jumps) { jumps.clear(); });
  for (auto const spike : output) {
    post_spikes.at(spike.source_neuron).push_back(spike);
  }
}

void LIF::set_input_spikes(SpikeVector input) {
  input_spikes = input;
  n_spikes = static_cast<int>(input.size());
  input_times = extract_times(input_spikes);
  input_sources = extract_sources(input_spikes);
  exp_input_mem = Eigen::exp(input_times.array() / tau_mem);
  exp_input_syn = Eigen::exp(input_times.array() / tau_syn);
  // compute sums for tmax
  sum0 = Eigen::MatrixXd::Zero(n_spikes + 1, n);
  sum1 = Eigen::MatrixXd::Zero(n_spikes + 1, n);
  for (int i = 0; i < n_spikes; i++) {
    sum0.row(i + 1) = sum0.row(i).array() +
                      exp_input_syn[i] * w.row(input_sources[i]).array();
    sum1.row(i + 1) = sum1.row(i).array() +
                      exp_input_mem[i] * w.row(input_sources[i]).array();
  }
  input_initialized = true;
}

void LIF::set_weights(Eigen::MatrixXd const weights) { w = weights; }

void LIF::zero_grad() { gradient = Eigen::MatrixXd::Zero(w.rows(), w.cols()); }

LIF::LIF(unsigned long int layer_id, double v_th,
         double tau_mem, double tau_syn, Eigen::MatrixXd const w)
    : v_th(v_th), tau_mem(tau_mem), tau_syn(tau_syn), w(w),
      post_spikes(w.cols()), layer_id(layer_id),
      n_in(static_cast<int>(w.rows())), n(static_cast<int>(w.cols())),
      tmax_prefactor(1 / (1 / tau_mem - 1 / tau_syn)),
      tmax_summand(std::log(tau_syn / tau_mem)),
      k_prefactor(tau_syn / (tau_mem - tau_syn)),
      k_bwd_prefactor(tau_mem / tau_syn), lambda_i_jumps(w.cols()),
      ran_forward(w.cols(), false), ran_backward(w.cols(), false) {
  zero_grad();
}

PYBIND11_MODULE(lif_layer_cpp, m) {
  py::class_<Spike>(m, "Spike")
      .def(py::init<int, double, unsigned long int, double>(),
           py::arg("source_neuron"), py::arg("time"),
           py::arg("source_layer") = 0, py::arg("error") = 0)
      .def_readonly("source_neuron", &Spike::source_neuron)
      .def_readwrite("time", &Spike::time)
      .def_readonly("source_layer", &Spike::source_layer)
      .def_readwrite("error", &Spike::error)
      .def("__repr__",
           [](Spike const &spike) {
             return (boost::format("<Spike w/ source_neuron=%1%, time=%2%, "
                                   "source_layer=%3%, error=%4%>") %
                     spike.source_neuron % spike.time % spike.source_layer %
                     spike.error)
                 .str();
           })
      .def("__eq__",
           [](Spike const &a, Spike const &b) {
             return (a.time == b.time && a.source_neuron == b.source_neuron &&
                     a.source_layer == b.source_layer && a.error == b.error);
           })
      .def(py::pickle(
          [](Spike const &spike) { // __getstate__
            return py::make_tuple(spike.source_neuron, spike.time,
                                  spike.source_layer, spike.error);
          },
          [](py::tuple t) { // __setstate__
            if (t.size() != 4) {
              throw std::runtime_error("Invalid state!");
            }
            Spike spike(t[0].cast<py::int_>(), t[1].cast<py::float_>(),
                        t[2].cast<py::int_>(), t[3].cast<py::float_>());
            return spike;
          }));

  py::class_<LIF>(m, "LIF")
      .def(py::init<unsigned long int, double, double,
                    double, Eigen::MatrixXd const>(),
           py::arg("layer_id"), py::arg("v_th"), py::arg("tau_mem"),
           py::arg("tau_syn"), py::arg("w"))
      .def("set_input_spikes", &LIF::set_input_spikes)
      .def("set_post_spikes", &LIF::set_post_spikes)
      .def("set_weights", &LIF::set_weights)
      .def("zero_grad", &LIF::zero_grad)
      .def_readonly("tau_syn", &LIF::tau_syn)
      .def_readonly("tau_mem", &LIF::tau_mem)
      .def_readonly("gradient", &LIF::gradient)
      .def_readonly("post_spikes", &LIF::post_spikes)
      .def_readonly("input_spikes", &LIF::input_spikes)
      .def("get_spikes_for_neuron", &LIF::get_spikes_for_neuron)
      .def("get_spikes", &LIF::get_spikes)
      .def("get_errors_for_neuron", &LIF::get_errors_for_neuron)
      .def("get_errors", &LIF::get_errors)
      .def("v", &LIF::v)
      .def("i", &LIF::i)
      .def("lambda_v", &LIF::lambda_v)
      .def("lambda_i", &LIF::lambda_i)
      .def("get_voltage_trace", &LIF::get_voltage_trace,
           py::arg("target_nrn_idx"), py::arg("t_max"), py::arg("dt") = 1e-4)
      .def("get_lambda_i_trace", &LIF::get_lambda_i_trace,
           py::arg("target_nrn_idx"), py::arg("t_max"), py::arg("dt") = 1e-4);
};