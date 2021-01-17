#include <Eigen/Dense>
#include <limits>
#include <vector>

double inf = std::numeric_limits<double>::infinity();
double eps = std::numeric_limits<double>::epsilon();

class Spike {
public:
  Spike(int source_neuron, double time, unsigned long int source_layer = 0,
        double error = 0)
      : source_neuron(source_neuron), time(time), source_layer(source_layer),
        error(error) {}
  bool operator<(Spike const &other) { return time < other.time; }
  bool operator>(Spike const &other) { return time > other.time; }

  int source_neuron;
  double time;
  unsigned long int source_layer;
  double error;
};

typedef std::vector<Spike> SpikeVector;
typedef std::reference_wrapper<Spike> SpikeRef;
typedef std::vector<SpikeRef> SpikeRefVector;

struct LambdaJump {
  double const value;
  double const time;
};

typedef std::vector<LambdaJump> LambdaJumpVector;

class LIF {
public:
  LIF(unsigned long int, double, double, double,
      Eigen::MatrixXd const);
  void set_input_spikes(SpikeVector);
  void set_post_spikes(SpikeVector);
  void set_weights(Eigen::MatrixXd const);
  void zero_grad();
  void get_spikes_for_neuron(int);
  void get_spikes();
  void get_errors_for_neuron(int);
  void get_errors();
  double v(double, int) const;
  double i(double, int) const;
  double lambda_v(double, int) const;
  double lambda_i(double, int) const;
  std::vector<double> get_voltage_trace(int, double,
                                        double) const;
  std::vector<double> get_lambda_i_trace(int, double,
                                         double) const;

  double const v_th;
  double const tau_mem;
  double const tau_syn;
  Eigen::MatrixXd w;
  Eigen::MatrixXd gradient;
  SpikeVector input_spikes;
  std::vector<SpikeVector> post_spikes;

private:
  Eigen::VectorXd extract_times(SpikeVector const &) const;
  Eigen::VectorXi extract_sources(SpikeVector const &) const;
  double get_tmax(int, int) const;
  double inline v_delta(double, int) const;
  double inline k(double) const;
  double inline k_bwd(double) const;
  double bracket_spike(double, double, int) const;

  unsigned long int const layer_id;
  bool input_initialized = false;
  int const n_in;
  int const n;
  int n_spikes;
  double const tmax_prefactor;
  double const tmax_summand;
  double const k_prefactor;
  double const k_bwd_prefactor;
  Eigen::VectorXd input_times;
  Eigen::VectorXi input_sources;
  Eigen::VectorXd exp_input_mem;
  Eigen::VectorXd exp_input_syn;
  Eigen::MatrixXd sum0;
  Eigen::MatrixXd sum1;
  std::vector<LambdaJumpVector> lambda_i_jumps;
  std::vector<bool> ran_forward;
  std::vector<bool> ran_backward;
};