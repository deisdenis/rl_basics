import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
import argparse

from spaces import bandit
from rlmodels import rlmodels


def main(app_args):
    perf = []
    opt = []
    k = 10
    n_blocks = 2000
    n_trials = 1000
    epsilons = [0, 0.01, 0.1]
    for eps in epsilons:
        start_time = dt.datetime.now()
        eps_performance = np.zeros([n_trials])
        optimal_actions = np.zeros([n_trials])
        for block in range(n_blocks):
            k_bandit = bandit.Bandit(k=k)
            optimal_action = k_bandit.get_optimal_action()
            model = rlmodels.EpsilonGreedy([i for i in range(k)], epsilon=eps)
            if app_args.optimistic_start:
                model.set_q(np.repeat(5.0, k))
            for i in range(n_trials):
                a = model.get_action()
                if a == optimal_action:
                    optimal_actions[i] += 1
                reward = k_bandit.use_lever(a)
                model.process_result(reward)
                eps_performance[i] += reward
        eps_performance /= n_blocks
        optimal_actions /= n_blocks
        perf.append(eps_performance)
        opt.append(optimal_actions)
        end_time = dt.datetime.now()
        print(f'Eps {eps} batch processed in {(end_time - start_time).total_seconds()}')
    x = range(n_trials)
    plt.plot(x, perf[0], 'g-', x, perf[1], 'r-', x, perf[2], 'b-')
    plt.title('Average reward')
    plt.show()
    plt.plot(x, opt[0], 'g-', x, opt[1], 'r-', x, opt[2], 'b-')
    plt.title('% Optimal Action')
    plt.show()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-o", "--optimistic-start", help="optimistic initialization of q values",
                           required=False, action='store_true')
    args = argParser.parse_args()
    main(args)
