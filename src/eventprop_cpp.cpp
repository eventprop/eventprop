#include "eventprop_cpp.h"

#include <Eigen/Core>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <cstdio>
#include <functional>

using namespace pybind11::literals;

Spikes
compute_spikes(Eigen::Ref<RowMatrixXd const> w, Spikes const &spikes,
               double v_th, double tau_mem, double tau_syn) {
  // compute constants
  auto const k_prefactor = tau_syn / (tau_mem - tau_syn);
  auto const tmax_prefactor = 1 / (1 / tau_mem - 1 / tau_syn);
  auto const tmax_summand = std::log(tau_syn / tau_mem);

  auto const &times = spikes.times;
  auto const &sources = spikes.sources;
  Eigen::ArrayXd const exp_input_mem = Eigen::exp(times.array() / tau_mem);
  Eigen::ArrayXd const exp_input_syn = Eigen::exp(times.array() / tau_syn);
  auto const n_spikes = times.size();
  auto const n = w.cols();
  RowMatrixXd sum0 = RowMatrixXd::Zero(n_spikes + 1, n);
  RowMatrixXd sum1 = RowMatrixXd::Zero(n_spikes + 1, n);
  // compute sums for tmax
  for (int i = 0; i < n_spikes; i++) {
    sum0.row(i + 1) =
        sum0.row(i).array() + exp_input_syn[i] * w.row(sources[i]).array();
    sum1.row(i + 1) =
        sum1.row(i).array() + exp_input_mem[i] * w.row(sources[i]).array();
  }

  std::vector<std::vector<double>> post_spikes(n);
  // define functions to compute voltage and maxima
  auto v = [&](double t, int target_nrn_idx, int t_pre_idx) {
    auto mem = k_prefactor *
               (sum1(t_pre_idx, target_nrn_idx) * std::exp(-t / tau_mem) -
                sum0(t_pre_idx, target_nrn_idx) * std::exp(-t / tau_syn));
    for (auto time : post_spikes.at(target_nrn_idx)) {
      if (time > t)
        break;
      mem -= v_th * std::exp(-(t - time) / tau_mem);
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
    for (auto time : post_spikes.at(target_nrn_idx)) {
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
  auto bracket_spike = [&](double a, double b, int target_nrn_idx,
                           int t_pre_idx) {
    boost::math::tools::eps_tolerance<double> t(
        std::numeric_limits<double>::digits - 1);
    boost::uintmax_t max_iter = boost::math::policies::get_max_root_iterations<
        boost::math::policies::policy<>>();
    std::pair<double, double> result = boost::math::tools::toms748_solve(
        [&](double t) { return v_delta(t, target_nrn_idx, t_pre_idx); }, a, b,
        t, max_iter);
    return (result.first + result.second) / 2;
  };

#pragma omp parallel for
  for (int target_nrn_idx = 0; target_nrn_idx < n; target_nrn_idx++) {
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
              for (auto time : post_spikes.at(target_nrn_idx)) {
                if (time > t_pre)
                  break;
                if (t_before < time) {
                  t_before = time;
                }
              }
            }
            t_post = bracket_spike(t_before, tmax, target_nrn_idx, i);
            post_spikes.at(target_nrn_idx).push_back(t_post);
            processed_up_to = i;
            break;
          }
        }
        if (v(t_pre, target_nrn_idx, i) > v_th + eps) {
          t_before = times[i - 1];
          tmax = t_pre;
          t_post = bracket_spike(t_before, tmax, target_nrn_idx, i);
          post_spikes.at(target_nrn_idx).push_back(t_post);
          processed_up_to = i;
          break;
        }
        if (i == n_spikes) {
          finished = true;
        }
      }
    }
  }
  std::vector<double> first_spike_times;
  std::transform(post_spikes.begin(), post_spikes.end(),
                 std::back_inserter(first_spike_times),
                 [](std::vector<double> spikes) -> double {
                   if (spikes.empty()) {
                     return NAN;
                   } else {
                     return spikes.front();
                   }
                 });
  std::vector<std::pair<double, int>> all_post_spikes;
  for (int nrn_idx = 0; nrn_idx < n; nrn_idx++) {
    std::transform(post_spikes.at(nrn_idx).begin(),
                   post_spikes.at(nrn_idx).end(),
                   std::back_inserter(all_post_spikes),
                   [&](double t) -> std::pair<double, int> {
                     return {t, nrn_idx};
                   });
  }
  std::sort(all_post_spikes.begin(), all_post_spikes.end(),
            [](std::pair<double, int> a, std::pair<double, int> b) {
              return a.first < b.first;
            });
  auto all_times_array = Eigen::ArrayXd(all_post_spikes.size());
  auto all_sources_array = Eigen::ArrayXi(all_post_spikes.size());
  std::vector<int> first_spike_idxs(n);
  #pragma omp parallel for
  for (int nrn_idx=0; nrn_idx<n; nrn_idx++) {
    if (not std::isnan(first_spike_times.at(nrn_idx))) {
      auto find_result = std::distance(all_post_spikes.begin(), std::find_if(all_post_spikes.begin(), all_post_spikes.end(), [&](std::pair<double, int> spike) -> bool { return spike.second == nrn_idx; }));
      first_spike_idxs.at(nrn_idx) = find_result;
    }
  }
#pragma omp parallel for
  for (int i = 0; i < all_post_spikes.size(); i++) {
    all_times_array(i) = all_post_spikes.at(i).first;
    all_sources_array(i) = all_post_spikes.at(i).second;
  }
  return {all_times_array, all_sources_array, first_spike_times, first_spike_idxs};
}

std::vector<Spikes>
compute_spikes_batch(Eigen::Ref<RowMatrixXd const> w,
                     std::vector<Spikes> const &batch, double v_th,
                     double tau_mem, double tau_syn) {
  auto result = std::vector<Spikes>(batch.size());
#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < batch.size(); batch_idx++) {
    result.at(batch_idx) =
        compute_spikes(w, batch[batch_idx], v_th, tau_mem, tau_syn);
  }
  return result;
}

void backward(Spikes &input_spikes, Spikes const &post_spikes,
              Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient,
              double v_th, double tau_mem, double tau_syn) {
  auto const n = gradient.cols();
  auto const n_input_spikes = input_spikes.times.size();

  std::vector<int> input_idxs(input_spikes.times.size());
  std::vector<int> post_idxs(post_spikes.times.size());
  std::vector<std::pair<int, bool>> sorted_idxs;
  std::iota(input_idxs.begin(), input_idxs.end(), 0);
  std::iota(post_idxs.begin(), post_idxs.end(), 0);
  std::for_each(input_idxs.begin(), input_idxs.end(), [&](int idx) {
    sorted_idxs.push_back({idx, true});
  });
  std::for_each(post_idxs.begin(), post_idxs.end(), [&](int idx) {
    sorted_idxs.push_back({idx, false});
  });
  std::sort(sorted_idxs.begin(), sorted_idxs.end(),
            [&](std::pair<int, bool> a, std::pair<int, bool> b) -> bool {
              double t_a, t_b;
              if (a.second)
                t_a = input_spikes.times[a.first];
              else
                t_a = post_spikes.times[a.first];
              if (b.second)
                t_b = input_spikes.times[b.first];
              else
                t_b = post_spikes.times[b.first];
              return t_a > t_b;
            });
  auto get_time = [&](std::pair<int, bool> idx) {
    if (idx.second)
      return input_spikes.times[idx.first];
    else
      return post_spikes.times[idx.first];
  };
  auto get_source = [&](std::pair<int, bool> idx) {
    if (idx.second)
      return input_spikes.sources[idx.first];
    else
      return post_spikes.sources[idx.first];
  };
  Eigen::ArrayXd lambda_v = Eigen::ArrayXd::Zero(n);
  std::vector<std::vector<std::pair<double, double>>> lambda_i_jumps(n);
  auto const largest_time = get_time(sorted_idxs.front());
  double previous_t = largest_time;

  auto const k_bwd_prefactor = tau_mem / (tau_mem - tau_syn);
  auto k_bwd = [&](double t) {
    return k_bwd_prefactor * (std::exp(-t / tau_mem) - std::exp(-t / tau_syn));
  };
  auto get_i = [&](double t, int target_nrn_idx) {
    double current = 0;
    for (int idx = 0; idx < n_input_spikes; idx++) {
      if (input_spikes.times[idx] > t)
        break;
      current += w(input_spikes.sources[idx], target_nrn_idx) *
                 std::exp(-(t - input_spikes.times[idx]) / tau_syn);
    }
    return current;
  };
  for (auto const &spike_idx : sorted_idxs) {
    auto const spike_time = get_time(spike_idx);
    auto const spike_source = get_source(spike_idx);
    auto const t_bwd = largest_time - spike_time;
    lambda_v = lambda_v * std::exp(-(t_bwd - previous_t) / tau_mem);
    if (spike_idx.second) {
      Eigen::ArrayXd lambda_i = Eigen::ArrayXd::Zero(n);
      for (int nrn_idx = 0; nrn_idx < n; nrn_idx++) {
        for (auto const &jump : lambda_i_jumps.at(nrn_idx)) {
          lambda_i[nrn_idx] += jump.first * k_bwd(t_bwd - jump.second);
        }
#pragma omp critical
        gradient(spike_source, nrn_idx) += -tau_syn * lambda_i[nrn_idx];
      }
      auto const outbound_signal =
          (w.row(spike_source).array().transpose() * (lambda_v - lambda_i))
              .sum();
#pragma omp critical
      input_spikes.errors[spike_idx.first] += outbound_signal;
    } else {
      auto const i = get_i(spike_time, spike_source);
      lambda_i_jumps.at(spike_source)
          .push_back({1 / (i - v_th) *
                          (v_th * lambda_v[spike_source] +
                           post_spikes.errors[spike_idx.first]),
                      t_bwd});
      lambda_v[spike_source] =
          i / (i - v_th) * lambda_v[spike_source] +
          1 / (i - v_th) * post_spikes.errors[spike_idx.first];
    }
    previous_t = t_bwd;
  }
}

void backward_batch(std::vector<Spikes> &input_batch,
                    std::vector<Spikes> const &post_batch,
                    Eigen::Ref<RowMatrixXd const> w,
                    Eigen::Ref<RowMatrixXd> gradient, double v_th,
                    double tau_mem, double tau_syn) {
  auto const n_batch = input_batch.size();
#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < n_batch; batch_idx++) {
    backward(input_batch[batch_idx], post_batch[batch_idx], w, gradient, v_th,
             tau_mem, tau_syn);
  }
}

PYBIND11_MODULE(eventprop_cpp, m) {
  py::class_<Spikes>(m, "Spikes")
      .def(py::init<Eigen::ArrayXd, Eigen::ArrayXi>(), "times"_a.noconvert(),
           "sources"_a.noconvert())
      .def(py::init<Eigen::ArrayXd, Eigen::ArrayXi, std::vector<double>, std::vector<int>>(), "times"_a.noconvert(),
           "sources"_a.noconvert(), "first_spike_times"_a, "first_spike_idxs"_a)
      .def("set_error", &Spikes::set_error, "spike_idx"_a, "error"_a)
      .def("set_time", &Spikes::set_time, "spike_idx"_a, "time"_a)
      .def_readonly("times", &Spikes::times)
      .def_readonly("sources", &Spikes::sources)
      .def_readonly("errors", &Spikes::errors)
      .def_readonly("n_spikes", &Spikes::n_spikes)
      .def_readonly("first_spike_times", &Spikes::first_spike_times)
      .def_readonly("first_spike_idxs", &Spikes::first_spike_idxs)
      .def(py::pickle(
          [](const Spikes &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.times, p.sources);
          },
          [](py::tuple t) { // __setstate__
            if (t.size() != 2)
              throw std::runtime_error("Invalid state!");
            /* Create a new C++ instance */
            Spikes p(t[0].cast<Eigen::ArrayXd>(), t[1].cast<Eigen::ArrayXi>());
            return p;
          }));
  py::bind_vector<std::vector<Spikes>>(m, "SpikesVector");
  m.def("compute_spikes_batch_cpp", &compute_spikes_batch, "w"_a.noconvert(),
        "batch"_a, "v_th"_a, "tau_mem"_a, "tau_syn"_a)
      .def("backward_batch_cpp", &backward_batch, "input_batch"_a.noconvert(),
           "post_batch"_a.noconvert(), "w"_a.noconvert(),
           "gradient"_a.noconvert(), "v_th"_a, "tau_mem"_a, "tau_syn"_a);
};