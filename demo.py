import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from agent import Agent
from tools.utility import smooth_ployline


def interaction(av, bv):
    av.estimated_inter_agent = [copy.deepcopy(bv)]

    av.lp_ibr_interact(iter_limit=50, interactive=True)
    # av.ibr_interact(iter_limit=50)

    av_solution = av.trj_solution[:, 0:5]
    bv_solution = av.estimated_inter_agent[0].trj_solution[:, 0:5]
    return av_solution, bv_solution


if __name__ == '__main__':
    cv, _ = smooth_ployline(np.array([[0, -1],
                                      [300, -1],
                                      [600, -1],
                                      [900, -1]]))

    fig, axes = plt.subplots(3, 1, figsize=[15, 15])

    for i, ipv in enumerate([-math.pi / 4, 0, math.pi / 4]):
        # background vehicle
        agent_bv = Agent(
            [0, -1],  # position
            [2, 0],  # velocity
            0,  # heading
            cv  # central vertices of the target lane
        )
        # agent_bv.ipv = math.pi / 8

        agent_av = Agent(
            [3, 1],  # position
            [2, 0],  # velocity
            0,  # heading
            cv  # central vertices of the target lane
        )
        agent_av.ipv = ipv
        av_sol, bv_sol = interaction(agent_av, agent_bv)
        dis = np.linalg.norm(av_sol-bv_sol, axis=1)

        axes[i].plot([0, 50], [-1, -1], color='gray', alpha=0.5)
        axes[i].set_xlim([0, 30])
        axes[i].set_ylim([-2, 2])

        axes[i].scatter(av_sol[:, 0], av_sol[:, 1])
        axes[i].scatter(bv_sol[:, 0], bv_sol[:, 1])

    plt.show()
