#include "eventprop_cpp.h"

#ifndef __APPLE__
#include <execution>
#endif
#include <Eigen/Core>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <cstdio>
#include <functional>

using namespace pybind11::literals;

std::pair<RowMatrixXd, RowMatrixXd> compute_sums(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double tau_mem, double tau_syn) {
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
  return {sum0, sum1};
}

std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_voltage_trace(double t_max, double dt, int target_nrn_idx, Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double v_th, double tau_mem, double tau_syn) {
  auto const k_prefactor = tau_syn / (tau_mem - tau_syn);
  auto const post_spikes = compute_spikes(w, spikes, v_th, tau_mem, tau_syn);
  auto v = [&](double t) {
    double mem = 0;
    for (int spike_idx=0; spike_idx<spikes.n_spikes; spike_idx++) {
      if (spikes.times[spike_idx] > t)
        break;
      mem += w(spikes.sources[spike_idx], target_nrn_idx) * k_prefactor * (std::exp(-(t-spikes.times[spike_idx])/tau_mem) - std::exp(-(t-spikes.times[spike_idx])/tau_syn));
    }
    for (int spike_idx=0; spike_idx<post_spikes.n_spikes; spike_idx++) {
      if (post_spikes.times[spike_idx] > t)
        break;
      if (post_spikes.sources[spike_idx] == target_nrn_idx)
        mem -= v_th * std::exp(-(t - post_spikes.times[spike_idx]) / tau_mem);
    }
    return mem;
  };
  int const n_t = static_cast<int>(t_max/dt);
  Eigen::ArrayXd v_trace = Eigen::ArrayXd::Zero(n_t);
  Eigen::ArrayXd ts = Eigen::ArrayXd::Zero(n_t);
  for (int t_idx=0; t_idx<n_t; t_idx++) {
    v_trace[t_idx] = v(dt*t_idx);
    ts[t_idx] = dt*t_idx;
  }
  return {ts, v_trace};
}

double compute_lambda_i(double t, int target_nrn_idx, Spikes const& post_spikes, double v_th, double tau_mem, double tau_syn)
{
  auto const n = post_spikes.first_spike_times.size();
  Eigen::ArrayXd lambda_v = Eigen::ArrayXd::Zero(n);
  std::vector<std::vector<std::pair<double, double>>> lambda_i_jumps(n);
  auto const largest_time = post_spikes.times[post_spikes.n_spikes-1];
  double previous_t = largest_time;

  auto const k_bwd_prefactor = tau_mem / (tau_mem - tau_syn);
  auto k_bwd = [&](double t) {
    return k_bwd_prefactor * (std::exp(-t / tau_mem) - std::exp(-t / tau_syn));
  };

  for (int spike_idx=post_spikes.n_spikes-1; spike_idx>=0;spike_idx--) {
    auto const spike_time = post_spikes.times[spike_idx];
    if (spike_time < t)
      break;
    auto const spike_source = post_spikes.sources[spike_idx];
    auto const t_bwd = largest_time - spike_time;
    lambda_v = lambda_v * std::exp(-(t_bwd - previous_t) / tau_mem);
    auto const i = post_spikes.currents[spike_idx];
    lambda_i_jumps.at(spike_source)
        .push_back({1 / (i - v_th) *
                        (v_th * lambda_v[spike_source] +
                          post_spikes.errors[spike_idx]),
                    t_bwd});
    lambda_v[spike_source] =
        i / (i - v_th) * lambda_v[spike_source] +
        1 / (i - v_th) * post_spikes.errors[spike_idx];
    previous_t = t_bwd;
  }
  double lambda_i = 0;
  auto const t_bwd = largest_time - t;
  for (auto const &jump : lambda_i_jumps.at(target_nrn_idx)) {
    lambda_i += jump.first * k_bwd(t_bwd - jump.second);
  }
  return lambda_i;
}

std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_lambda_i_trace(double t_max, double dt, int target_nrn_idx, Spikes const& post_spikes, double v_th, double tau_mem, double tau_syn) {
  int const n_t = static_cast<int>(t_max/dt);
  Eigen::ArrayXd lambda_trace = Eigen::ArrayXd::Zero(n_t);
  Eigen::ArrayXd ts = Eigen::ArrayXd::Zero(n_t);
  for (int t_idx=0; t_idx<n_t; t_idx++) {
    lambda_trace[t_idx] = compute_lambda_i(dt*t_idx, target_nrn_idx, post_spikes, v_th, tau_mem, tau_syn);
    ts[t_idx] = dt*t_idx;
  }
  return {ts, lambda_trace};
}

Spikes
compute_spikes(Eigen::Ref<RowMatrixXd const> w, Spikes const &spikes,
               double v_th, double tau_mem, double tau_syn) {
  // compute constants
  auto const k_prefactor = tau_syn / (tau_mem - tau_syn);
  auto const tmax_prefactor = 1 / (1 / tau_mem - 1 / tau_syn);
  auto const tmax_summand = std::log(tau_syn / tau_mem);
  auto const &times = spikes.times;
  auto const &sources = spikes.sources;
  auto const n_spikes = times.size();
  auto const n = w.cols();
  RowMatrixXd sum0, sum1;
  std::tie(sum0, sum1) = compute_sums(w, spikes, tau_mem, tau_syn);

  std::vector<std::vector<std::pair<double, double>>> post_spikes(n);
  // define functions to compute voltage and maxima
  auto v = [&](double t, int target_nrn_idx, int t_pre_idx) {
    auto mem = k_prefactor *
               (sum1(t_pre_idx, target_nrn_idx) * std::exp(-t / tau_mem) -
                sum0(t_pre_idx, target_nrn_idx) * std::exp(-t / tau_syn));
    for (auto spike : post_spikes.at(target_nrn_idx)) {
      if (spike.first > t)
        break;
      mem -= v_th * std::exp(-(t - spike.first) / tau_mem);
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
    for (auto spike : post_spikes.at(target_nrn_idx)) {
      if (t_input <= spike.first)
        break;
      sum1_elem += -v_th * std::exp(spike.first / tau_mem);
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
  auto get_i = [&](double t, int target_nrn_idx) {
  double current = 0;
  int idx;
  for (idx = 0; idx < times.size(); idx++) {
    if (times[idx] > t)
      break;
  }
  current = sum0(idx, target_nrn_idx)*std::exp(-t/tau_syn);
  return current;
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
              for (auto spike : post_spikes.at(target_nrn_idx)) {
                if (spike.first > t_pre)
                  break;
                if (t_before < spike.first) {
                  t_before = spike.first;
                }
              }
            }
            t_post = bracket_spike(t_before, tmax, target_nrn_idx, i);
            post_spikes.at(target_nrn_idx).push_back({t_post, get_i(t_post, target_nrn_idx)});
            processed_up_to = i;
            break;
          }
        }
        if (v(t_pre, target_nrn_idx, i) > v_th + eps) {
          t_before = times[i - 1];
          tmax = t_pre;
          t_post = bracket_spike(t_before, tmax, target_nrn_idx, i);
          post_spikes.at(target_nrn_idx).push_back({t_post, get_i(t_post, target_nrn_idx)});
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
                 [&](std::vector<std::pair<double, double>> spikes) -> double {
                   if (spikes.empty()) {
                     return NAN;
                   } else {
                     return spikes.front().first;
                   }
                 });
  std::vector<std::pair<std::pair<double, double>, int>> all_post_spikes;
  for (int nrn_idx = 0; nrn_idx < n; nrn_idx++) {
    std::transform(post_spikes.at(nrn_idx).begin(),
                   post_spikes.at(nrn_idx).end(),
                   std::back_inserter(all_post_spikes),
                   [&](std::pair<double, double> spike) -> std::pair<std::pair<double, double>, int> {
                     return {spike, nrn_idx};
                   });
  }
  #ifdef __APPLE__
  std::sort(all_post_spikes.begin(), all_post_spikes.end(),
            [](std::pair<std::pair<double, double>, int> a, std::pair<std::pair<double, double>, int> b) {
              return a.first.first < b.first.first;
            });
  #else
  std::sort(std::execution::par_unseq, all_post_spikes.begin(), all_post_spikes.end(),
            [](std::pair<std::pair<double, double>, int> a, std::pair<std::pair<double, double>, int> b) {
              return a.first.first < b.first.first;
            });
  #endif
  auto all_times_array = Eigen::ArrayXd(all_post_spikes.size());
  auto all_sources_array = Eigen::ArrayXi(all_post_spikes.size());
  auto all_currents_array = Eigen::ArrayXd(all_post_spikes.size());
  std::vector<int> first_spike_idxs(n);
  #pragma omp parallel for
  for (int nrn_idx=0; nrn_idx<n; nrn_idx++) {
    if (not std::isnan(first_spike_times.at(nrn_idx))) {
      #ifdef __APPLE__
      auto find_result = std::distance(all_post_spikes.begin(), std::find_if(all_post_spikes.begin(), all_post_spikes.end(), [&](std::pair<std::pair<double, double>, int> spike) -> bool { return spike.second == nrn_idx; }));
      #else
      auto find_result = std::distance(all_post_spikes.begin(), std::find_if(std::execution::par_unseq, all_post_spikes.begin(), all_post_spikes.end(), [&](std::pair<std::pair<double, double>, int> spike) -> bool { return spike.second == nrn_idx; }));
      #endif

      first_spike_idxs.at(nrn_idx) = find_result;
    }
  }
#pragma omp parallel for
  for (int i = 0; i < all_post_spikes.size(); i++) {
    all_times_array(i) = all_post_spikes.at(i).first.first;
    all_sources_array(i) = all_post_spikes.at(i).second;
    all_currents_array(i) = all_post_spikes.at(i).first.second;
  }
  double n_dead = 0;
  for (auto t: first_spike_times) {
    if (std::isnan(t)) {
      n_dead += 1;
    }
  }
  return {all_times_array, all_sources_array, all_currents_array, first_spike_times, first_spike_idxs, n_dead/static_cast<double>(n)};
}

std::pair<std::vector<Spikes>, double>
compute_spikes_batch(Eigen::Ref<RowMatrixXd const> w,
                     std::vector<Spikes> const &batch, double v_th,
                     double tau_mem, double tau_syn) {
  auto result = std::vector<Spikes>(batch.size());
#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < batch.size(); batch_idx++) {
    result.at(batch_idx) =
        compute_spikes(w, batch[batch_idx], v_th, tau_mem, tau_syn);
  }
  double avg_dead_fraction = 0;
  for (auto const& spikes : result) {
    avg_dead_fraction += spikes.dead_fraction;
  }
  avg_dead_fraction /= (double)batch.size();
  return {result, avg_dead_fraction};
}

std::pair<RowMatrixXd, Eigen::ArrayXd> backward(Spikes &input_spikes, Spikes const &post_spikes,
              Eigen::Ref<RowMatrixXd const> w,
              double v_th, double tau_mem, double tau_syn) {
  RowMatrixXd gradient = RowMatrixXd::Zero(w.rows(), w.cols());
  Eigen::ArrayXd spike_errors = Eigen::ArrayXd::Zero(input_spikes.n_spikes);
  auto const n = gradient.cols();

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

  auto w_t = w.transpose();
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
        gradient(spike_source, nrn_idx) += -tau_syn * lambda_i[nrn_idx];
      }
      auto const outbound_signal =
          (w_t.col(spike_source).array() * (lambda_v - lambda_i))
              .sum();
      input_spikes.errors[spike_idx.first] += outbound_signal;
    } else {
      auto const i = post_spikes.currents[spike_idx.first];
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
  return {gradient, spike_errors};
}

void backward_spikes_batch(std::vector<Spikes> &input_batch,
                    std::vector<Spikes> const &post_batch,
                    Eigen::Ref<RowMatrixXd const> w,
                    Eigen::Ref<RowMatrixXd> gradient, double v_th,
                    double tau_mem, double tau_syn) {
  auto const n_batch = input_batch.size();
  std::vector<std::pair<RowMatrixXd, Eigen::ArrayXd>> partial_grads(n_batch);
#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < n_batch; batch_idx++) {
    if (post_batch[batch_idx].n_spikes > 0) {
      partial_grads.at(batch_idx) = backward(input_batch[batch_idx], post_batch[batch_idx], w, v_th,
              tau_mem, tau_syn);
    }
  }
  for (int batch_idx=0; batch_idx<n_batch;batch_idx++) {
    if (post_batch[batch_idx].n_spikes == 0) {
      continue;
    }
    auto const partial_gradient = partial_grads[batch_idx].first;
    auto const partial_spike_errors = partial_grads[batch_idx].second;
    gradient += partial_gradient;
    for (int spike_idx=0; spike_idx<partial_spike_errors.size(); spike_idx++) {
      input_batch[batch_idx].errors[spike_idx] += partial_spike_errors[spike_idx];
    }
  }
}


Maxima compute_maxima(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double tau_mem, double tau_syn) {
// compute constants
  auto const k_prefactor = tau_syn / (tau_mem - tau_syn);
  auto const tmax_prefactor = 1 / (1 / tau_mem - 1 / tau_syn);
  auto const tmax_summand = std::log(tau_syn / tau_mem);
  auto const &times = spikes.times;
  auto const n_spikes = times.size();
  auto const n = w.cols();
  RowMatrixXd sum0, sum1;
  std::tie(sum0, sum1) = compute_sums(w, spikes, tau_mem, tau_syn);

  auto v = [&](double t, int target_nrn_idx, int t_pre_idx) {
    auto mem = k_prefactor *
               (sum1(t_pre_idx, target_nrn_idx) * std::exp(-t / tau_mem) -
                sum0(t_pre_idx, target_nrn_idx) * std::exp(-t / tau_syn));
    return mem;
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
    double tmax =
        tmax_prefactor * (tmax_summand + std::log(sum1_elem / sum0_elem));
    if (t_before <= tmax && tmax <= t_input) {
      return tmax;
    }
    return inf;
  };

  Eigen::ArrayXd maxima_values = Eigen::ArrayXd::Zero(n);
  Eigen::ArrayXd maxima_times = Eigen::ArrayXd(n);
  for (int i=0; i<n; i++) {
    maxima_times[i] = NAN;
  }
#pragma omp parallel for
  for (int target_nrn_idx = 0; target_nrn_idx < n; target_nrn_idx++) {
    double tmax, vmax, t_pre;
      for (int i = 0; i < n_spikes + 1; i++) {
        if (i == n_spikes) {
          t_pre = inf;
        } else {
          t_pre = times[i];
        }
        tmax = get_tmax(i, target_nrn_idx);
        if (tmax != inf) {
          vmax = v(tmax, target_nrn_idx, i);
          if (vmax > maxima_values(target_nrn_idx)) {
            maxima_values(target_nrn_idx) = vmax;
            maxima_times(target_nrn_idx) = tmax;
          }
        }
        vmax = v(t_pre, target_nrn_idx, i);
        if (vmax > maxima_values(target_nrn_idx)) {
            maxima_values(target_nrn_idx) = vmax;
            maxima_times(target_nrn_idx) = t_pre;
        }
      }
  }
  return {maxima_times, maxima_values};
}

std::pair<RowMatrixXd, Eigen::ArrayXd> backward(Spikes & input_spikes, Maxima const& maxima, Eigen::Ref<RowMatrixXd const> w, double tau_mem, double tau_syn) {
  RowMatrixXd gradient = RowMatrixXd::Zero(w.rows(), w.cols());
  Eigen::ArrayXd spike_errors = Eigen::ArrayXd::Zero(input_spikes.n_spikes);
  auto const largest_time = input_spikes.times(input_spikes.n_spikes-1);
  auto const n_maxima = maxima.times.size();
  auto const k_bwd_prefactor = tau_mem/(tau_mem-tau_syn);
  auto k_bwd = [&](double t) { return k_bwd_prefactor*(std::exp(-t/tau_mem)-std::exp(-t/tau_syn));};
  for (int spike_idx = input_spikes.n_spikes-1; spike_idx >=0; spike_idx--) {
    auto const t_bwd = largest_time - input_spikes.times(spike_idx);
    for (int max_idx = 0; max_idx < n_maxima; max_idx++) {
      if (std::isnan(maxima.times(max_idx))) {
        continue;
      }
      auto const t_vmax_bwd = largest_time - maxima.times(max_idx);
      if (t_vmax_bwd > t_bwd) {
        continue;
      }
      auto const lambda_v = -std::exp(-(t_bwd-t_vmax_bwd)/tau_mem)*maxima.errors(max_idx)/tau_mem;
      auto const lambda_i = -k_bwd(t_bwd-t_vmax_bwd)*maxima.errors(max_idx)/tau_mem;
      gradient(input_spikes.sources(spike_idx), max_idx) += -tau_syn*lambda_i;
      spike_errors[spike_idx] += w(input_spikes.sources(spike_idx), max_idx) * (lambda_v - lambda_i);
    }
  }
  return {gradient, spike_errors};
}

std::vector<Maxima>
compute_maxima_batch(Eigen::Ref<RowMatrixXd const> w, std::vector<Spikes> const& batch, double tau_mem, double tau_syn) {
  auto result = std::vector<Maxima>(batch.size());
  #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch.size(); batch_idx++) {
    result.at(batch_idx) =
        compute_maxima(w, batch[batch_idx], tau_mem, tau_syn);
  }
  return result;
}

void
backward_maxima_batch(std::vector<Spikes> &input_batch, std::vector<Maxima> const& maxima, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double tau_mem, double tau_syn) {
  auto const n_batch = input_batch.size();
  std::vector<std::pair<RowMatrixXd, Eigen::ArrayXd>> partial_grads(n_batch);
#pragma omp parallel for
  for (int batch_idx = 0; batch_idx < n_batch; batch_idx++) {
    if (input_batch[batch_idx].n_spikes > 0) {
      partial_grads.at(batch_idx) = backward(input_batch[batch_idx], maxima[batch_idx], w,
              tau_mem, tau_syn);
    }
  }
  for (int batch_idx=0; batch_idx<n_batch;batch_idx++) {
    if (input_batch[batch_idx].n_spikes == 0) {
      continue;
    }
    auto const partial_gradient = partial_grads[batch_idx].first;
    auto const partial_spike_errors = partial_grads[batch_idx].second;
    gradient += partial_gradient;
    for (int spike_idx=0; spike_idx<partial_spike_errors.size(); spike_idx++) {
      input_batch[batch_idx].errors[spike_idx] += partial_spike_errors[spike_idx];
    }
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
  py::class_<Maxima>(m, "Maxima")
    .def(py::init<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd>(), "times"_a, "values"_a, "errors"_a)
    .def(py::init<Eigen::ArrayXd, Eigen::ArrayXd>(), "times"_a, "values"_a)
    .def("set_error", &Maxima::set_error, "nrn_idx"_a, "error"_a)
    .def("set_value", &Maxima::set_value, "nrn_idx"_a, "value"_a)
    .def_readonly("times", &Maxima::times)
    .def_readonly("values", &Maxima::values)
    .def_readonly("errors", &Maxima::errors);
  py::bind_vector<std::vector<Spikes>>(m, "SpikesVector")
  .def(py::pickle(
          [](std::vector<Spikes> const&vec) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            auto ret = py::list();
            for (auto const & spikes: vec) {
              ret.append(py::make_tuple(spikes.times, spikes.sources));
            }
            return ret;
          },
          [](py::list ret) { // __setstate__
            /* Create a new C++ instance */
            std::vector<Spikes> vec;
            for (int i=0; i<ret.size(); i++) {
              auto tup = ret[i].cast<py::tuple>();
              vec.push_back(Spikes(tup[0].cast<Eigen::ArrayXd>(), tup[1].cast<Eigen::ArrayXi>()));
            }
            return vec;
          }));
  py::bind_vector<std::vector<Maxima>>(m, "MaximaVector");
  m.def("compute_spikes_batch_cpp", &compute_spikes_batch, "w"_a.noconvert(),
        "batch"_a, "v_th"_a, "tau_mem"_a, "tau_syn"_a)
      .def("backward_spikes_batch_cpp", &backward_spikes_batch, "input_batch"_a.noconvert(),
           "post_batch"_a.noconvert(), "w"_a.noconvert(),
           "gradient"_a.noconvert(), "v_th"_a, "tau_mem"_a, "tau_syn"_a)
  .def("compute_maxima_batch_cpp", &compute_maxima_batch, "w"_a.noconvert(),
        "batch"_a, "tau_mem"_a, "tau_syn"_a)
      .def("backward_maxima_batch_cpp", &backward_maxima_batch, "input_batch"_a.noconvert(),
           "maxima"_a.noconvert(), "w"_a.noconvert(),
           "gradient"_a.noconvert(), "tau_mem"_a, "tau_syn"_a)
  .def("compute_voltage_trace_cpp", &compute_voltage_trace, "t_max"_a, "dt"_a, "target_nrn_idx"_a, "w"_a.noconvert(), "spikes"_a, "v_th"_a, "tau_mem"_a, "tau_syn"_a)
  .def("compute_lambda_i_cpp", &compute_lambda_i, "t"_a, "target_nrn_idx"_a, "post_spikes"_a, "v_th"_a, "tau_mem"_a, "tau_syn"_a)
  .def("compute_lambda_i_trace_cpp", &compute_lambda_i_trace, "t_max"_a, "dt"_a, "target_nrn_idx"_a, "post_spikes"_a, "v_th"_a, "tau_mem"_a, "tau_syn"_a);
};