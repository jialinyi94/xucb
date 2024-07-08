#include <vector>
#include <cmath>
#include <algorithm>

class UCB {
private:
    int n_arms;
    std::vector<double> values;
    std::vector<int> counts;
    int total_count;

public:
    UCB(int n_arms) : n_arms(n_arms), values(n_arms, 0.0), counts(n_arms, 0), total_count(0) {}

    int select_arm() {
        for (int i = 0; i < n_arms; i++) {
            if (counts[i] == 0) return i;
        }

        std::vector<double> ucb_values(n_arms);
        for (int i = 0; i < n_arms; i++) {
            double bonus = std::sqrt(2.0 * std::log(total_count) / counts[i]);
            ucb_values[i] = values[i] + bonus;
        }

        return std::max_element(ucb_values.begin(), ucb_values.end()) - ucb_values.begin();
    }

    void update(int chosen_arm, double reward) {
        counts[chosen_arm]++;
        total_count++;
        double n = counts[chosen_arm];
        double value = values[chosen_arm];
        values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward;
    }

    std::vector<int> multi_step(int num_steps, const double* rewards) {
        std::vector<int> chosen_arms;
        for (int i = 0; i < num_steps; i++) {
            int arm = select_arm();
            update(arm, rewards[arm]);
            chosen_arms.push_back(arm);
        }
        return chosen_arms;
    }

    std::vector<double> get_values() const {
        return values;
    }

    std::vector<int> get_counts() const {
        return counts;
    }
};
