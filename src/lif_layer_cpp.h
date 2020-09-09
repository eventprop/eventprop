#include <vector>
#include <Eigen/Dense>
#include <limits>

double inf = std::numeric_limits<double>::infinity();
double eps = std::numeric_limits<double>::epsilon();

class Spike {
    public:
        Spike(int source_neuron, double time, unsigned long int source_layer = 0, double error = 0) : source_neuron(source_neuron), time(time), source_layer(source_layer), error(error) { }
        bool operator<(Spike const& other) { return time < other.time; }
        bool operator>(Spike const& other) { return time > other.time; }

        int source_neuron;
        double time;
        unsigned long int source_layer;
        double error;
};

typedef std::vector<Spike> SpikeVector;
typedef std::reference_wrapper<Spike> SpikeRef;
typedef std::vector<SpikeRef> SpikeRefVector;

class LIF {
    public:
        LIF(unsigned long int const, double const, double const, double const, Eigen::MatrixXd  const);
        void set_input_spikes(SpikeVector);
        void set_post_spikes(SpikeVector);
        void set_weights(Eigen::MatrixXd const);
        void zero_grad();
        void get_spikes_for_neuron(int const);
        void get_spikes();
        void get_errors_for_neuron(int const);
        void get_errors();
        double const v(double const, int const) const;
        double const i(double const, int const) const;
        double const lambda_v(double const, int const) const;
        double const lambda_i(double const, int const) const;
        std::vector<double> get_voltage_trace(int const, double const, double const) const;
        std::vector<double> get_lambda_i_trace(int const, double const, double const) const;

        double const v_th;
        double const tau_mem;
        double const tau_syn;
        Eigen::MatrixXd w;
        Eigen::MatrixXd gradient;
        SpikeVector input_spikes;
        std::vector<SpikeVector> post_spikes;

    private:
        Eigen::VectorXd extract_times(SpikeVector const&) const;
        Eigen::VectorXi extract_sources(SpikeVector const&) const;
        double const get_tmax(int const, int const) const;
        double const inline v_delta(double const, int const) const;
        double const inline k(double const) const;
        double const inline k_bwd(double const) const;
        double const bracket_spike(double const, double const, int const) const;

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
        std::vector<std::vector<std::pair<double, double>>> lambda_i_spikes;
        std::vector<bool> ran_forward;
        std::vector<bool> ran_backward;
};