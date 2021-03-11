#include "lif_layer_cpp.h"

#include <algorithm>
#include <boost/format.hpp>
#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

using namespace pybind11::literals;

py::tuple compute_spikes(Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<Eigen::ArrayXd const> times, Eigen::Ref<Eigen::ArrayXi const> sources, double v_th, double tau_mem, double tau_syn) {
  // compute constants
  auto const k_prefactor = tau_syn/(tau_mem-tau_syn);
  auto const tmax_prefactor = 1/(1/tau_mem-1/tau_syn);
  auto const tmax_summand = std::log(tau_syn/tau_mem);

  Eigen::ArrayXd const exp_input_mem = Eigen::exp(times.array() / tau_mem);
  Eigen::ArrayXd const exp_input_syn = Eigen::exp(times.array() / tau_syn);
  auto const n_spikes = times.size();
  auto const n = w.cols();
  RowMatrixXd sum0 = RowMatrixXd::Zero(n_spikes+1, n);
  RowMatrixXd sum1 = RowMatrixXd::Zero(n_spikes+1, n);
  // compute sums for tmax
  for (int i = 0; i < n_spikes; i++) {
    sum0.row(i + 1) = sum0.row(i).array() + exp_input_syn[i] * w.row(sources[i]).array();
    sum1.row(i + 1) = sum1.row(i).array() + exp_input_mem[i] * w.row(sources[i]).array();
  }

  std::vector<std::vector<double>> post_times(n);
  // define functions to compute voltage and maxima
  auto v = [&](double t, int target_nrn_idx, int t_pre_idx) {
    auto mem =  k_prefactor*(sum1(t_pre_idx, target_nrn_idx)*std::exp(-t/tau_mem) - sum0(t_pre_idx, target_nrn_idx)*std::exp(-t/tau_syn));
    for (auto time: post_times.at(target_nrn_idx)) {
      if (time > t) break;
      mem -= v_th*std::exp(-(t-time)/tau_mem);
    }
    return mem;
  };
  auto v_delta = [&](double t, int target_nrn_idx, int t_pre_idx) {
    return v(t, target_nrn_idx, t_pre_idx) - v_th;
  };
  auto get_tmax = [&](int t_pre_idx, int target_nrn_idx) {
    double t_input;
    if (t_pre_idx == 0) {
      return 0.0;
    } else if (t_pre_idx == n_spikes) {
      t_input = inf;
    } else {
      t_input = times[t_pre_idx];
    }
    double sum0_elem = sum0(t_pre_idx, target_nrn_idx);
    double sum1_elem = sum1(t_pre_idx, target_nrn_idx);
    double t_before = times[t_pre_idx - 1];
    for (auto time : post_times.at(target_nrn_idx)) {
      if (t_input <= time)
        break;
      sum1_elem += -v_th * std::exp(time / tau_mem);
    }
    double tmax =
        tmax_prefactor * (tmax_summand + std::log(sum1_elem / sum0_elem));
    if (t_before <= tmax && tmax <= t_input) {
      return tmax;
    }
    return inf;
  };
  auto bracket_spike = [&](double a, double b,
                                  int target_nrn_idx, int t_pre_idx) {
    boost::math::tools::eps_tolerance<double> t(
        std::numeric_limits<double>::digits - 1);
    boost::uintmax_t max_iter = boost::math::policies::get_max_root_iterations<
        boost::math::policies::policy<>>();
    std::pair<double, double> result = boost::math::tools::toms748_solve(
        [&](double t) { return v_delta(t, target_nrn_idx, t_pre_idx); }, a, b, t, max_iter);
    return (result.first + result.second) / 2;
  };

  std::vector<std::pair<double, int>> all_post_spikes;
  #pragma omp parallel for
  for (int target_nrn_idx=0; target_nrn_idx<n; target_nrn_idx++) {
    bool finished = false;
    int processed_up_to = 0;
    double tmax, vmax, t_before, t_pre, t_post;
    while (not finished) {
      for (int i = processed_up_to; i < n_spikes + 1; i++) {
        if (i == n_spikes) {
          t_pre = inf;
        } else {
          t_pre = times[i];
        }
        tmax = get_tmax(i, target_nrn_idx);
        if (tmax != inf) {
          vmax = v(tmax, target_nrn_idx, i);
          if (vmax > v_th + eps) {
            t_before = 0;
            if (i > 0) {
              t_before = times[i - 1];
              for (auto time : post_times.at(target_nrn_idx)) {
                if (time > t_pre)
                  break;
                if (t_before < time) {
                  t_before = time;
                }
              }
            }
            t_post = bracket_spike(t_before, tmax, target_nrn_idx, i);
            post_times.at(target_nrn_idx).push_back(t_post);
            #pragma omp critical
            all_post_spikes.push_back({t_post, target_nrn_idx});
            processed_up_to = i;
            break;
          }
        }
        if (v(t_pre, target_nrn_idx, i) > v_th + eps) {
          t_before = times[i - 1];
          tmax = t_pre;
          t_post = bracket_spike(t_before, tmax, target_nrn_idx, i);
          post_times.at(target_nrn_idx).push_back(t_post);
          #pragma omp critical
          all_post_spikes.push_back({t_post, target_nrn_idx});
          processed_up_to = i;
          break;
        }
        if (i == n_spikes) {
          finished = true;
        }
      }
    }
  }
  Eigen::ArrayXd all_times_array(all_post_spikes.size());
  Eigen::ArrayXi all_sources_array(all_post_spikes.size());
  std::sort(all_post_spikes.begin(), all_post_spikes.end(), [](std::pair<double, int> a, std::pair<double, int> b) { return a.first < b.first; });
  #pragma omp parallel for
  for (size_t i=0; i<all_post_spikes.size(); i++) {
    all_times_array[i] = all_post_spikes.at(i).first;
    all_sources_array[i] = all_post_spikes.at(i).second;
  }
  return py::make_tuple(all_times_array, all_sources_array);
}

void backward(Eigen::Ref<Eigen::ArrayXd const> input_times, Eigen::Ref<Eigen::ArrayXi const> input_sources, Eigen::Ref<Eigen::ArrayXd const> post_times, Eigen::Ref<Eigen::ArrayXi const> post_sources, Eigen::Ref<Eigen::ArrayXd> input_errors, Eigen::Ref<Eigen::ArrayXd const> post_errors, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double v_th, double tau_mem, double tau_syn) {
  auto const n = gradient.cols();

  std::vector<int> input_idxs(input_times.size());
  std::vector<int> post_idxs(post_times.size());
  std::vector<std::pair<int, bool>> sorted_idxs;
  std::iota(input_idxs.begin(), input_idxs.end(), 0);
  std::iota(post_idxs.begin(), post_idxs.end(), 0);
  std::for_each(input_idxs.begin(), input_idxs.end(), [&](int idx) {sorted_idxs.push_back({idx, true}); });
  std::for_each(post_idxs.begin(), post_idxs.end(), [&](int idx) {sorted_idxs.push_back({idx, false}); });
  std::sort(sorted_idxs.begin(), sorted_idxs.end(), [&](std::pair<int, bool> a, std::pair<int, bool> b) -> bool {
    double t_a, t_b;
    if (a.second)
      t_a = input_times[a.first];
    else
      t_a = post_times[a.first];
    if (b.second)
      t_b = input_times[b.first];
    else
      t_b = post_times[b.first];
    return t_a > t_b;
  });
  auto get_time = [&](std::pair<int, bool> idx) {
    if (idx.second)
      return input_times[idx.first];
    else
      return post_times[idx.first];
  };
  auto get_source = [&](std::pair<int, bool> idx) {
    if (idx.second)
      return input_sources[idx.first];
    else
      return post_sources[idx.first];
  };
  Eigen::ArrayXd lambda_v = Eigen::ArrayXd::Zero(n);
  std::vector<std::vector<std::pair<double, double>>> lambda_i_jumps(n);
  auto const largest_time = get_time(sorted_idxs.front());
  double previous_t = largest_time;

  auto k_bwd = [&](double t) { return tau_mem/(tau_mem-tau_syn)*(std::exp(-t/tau_mem)-std::exp(-t/tau_syn)); };
  auto get_i = [&](double t, int target_nrn_idx) {
    double current = 0;
    for (int idx=0; idx<input_times.size(); idx++) {
      if (input_times[idx] > t)
        break;
      current += w(input_sources[idx], target_nrn_idx)*std::exp(-(t-input_times[idx])/tau_syn);
    }
    return current;
  };
  for (auto const & spike_idx : sorted_idxs) {
    auto const spike_time = get_time(spike_idx);
    auto const spike_source = get_source(spike_idx);
    auto const t_bwd = largest_time - spike_time;
    lambda_v = lambda_v * std::exp(-(t_bwd-previous_t)/tau_mem);
    if (spike_idx.second) {
      Eigen::ArrayXd lambda_i = Eigen::ArrayXd::Zero(n);
      for (int nrn_idx=0; nrn_idx<n; nrn_idx++) {
        for (auto const& jump : lambda_i_jumps.at(nrn_idx)) {
          lambda_i[nrn_idx] += jump.first * k_bwd(t_bwd-jump.second);
        }
        gradient(spike_source, nrn_idx) += -tau_syn*lambda_i[nrn_idx];
      }
      auto const outbound_signal = (w.row(spike_source).array().transpose() * (lambda_v - lambda_i)).sum();
      input_errors[spike_idx.first] += outbound_signal;
    }
    else {
      auto const i = get_i(spike_time, spike_source);
      lambda_i_jumps.at(spike_source).push_back({1/(i-v_th)*(v_th*lambda_v[spike_source]+post_errors[spike_idx.first]), t_bwd});
      lambda_v[spike_source] = i/(i-v_th)*lambda_v[spike_source] + 1/(i-v_th)*post_errors[spike_idx.first];
    }
    previous_t = t_bwd;
  }
}

PYBIND11_MODULE(lif_layer_cpp, m) {
  m.def("compute_spikes_cpp", &compute_spikes, "w"_a.noconvert(), "times"_a.noconvert(), "sources"_a.noconvert(), "v_th"_a, "tau_mem"_a, "tau_syn"_a, py::return_value_policy::copy)
  .def("backward_cpp", &backward, "input_times"_a.noconvert(), "input_sources"_a.noconvert(), "post_times"_a.noconvert(), "post_sources"_a.noconvert(), "input_errors"_a.noconvert(), "post_errors"_a.noconvert(), "w"_a.noconvert(), "gradient"_a.noconvert(), "v_th"_a, "tau_mem"_a, "tau_syn"_a);
};