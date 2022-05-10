import sys

from typing import List


def write_to_file(lines: List[str], path: str):
    lines = [line + "\n" for line in lines]
    with open(path, 'w') as output_file:
        output_file.writelines(lines)


def get_action_lines(action):
    return [f"Name: {action[0]}",
            f"pre: {' '.join(action[1])}",
            f"add: {' '.join(action[2])}",
            f"delete: {' '.join(action[3])}"]


def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    elements = disks + pegs

    # disk is on an element (on disk or directly on peg)
    on_propositions = [f"{disk}_on_{element}" for disk in disks for element in elements]
    # is disk on top of some peg
    top_propositions = [f"{element}_top" for element in elements]
    # is disk smaller than other disk (disk always smaller than peg)
    smaller_than_propositions = [f"{disk}_st_{element}" for disk in disks for element in elements]
    propositions = on_propositions + top_propositions + smaller_than_propositions

    actions = []
    for disk in disks:
        for src in elements:
            for dest in elements:
                action_name = f"move_{disk}_from_{src}_to_{dest}"
                action_pre = [f"{disk}_top", f"{dest}_top", f"{disk}_on_{src}", f"{disk}_st_{dest}"]
                action_add = [f"{disk}_on_{dest}", f"{src}_top"]
                action_delete = [f"{dest}_top", f"{disk}_on_{src}"]
                actions.append((action_name, action_pre, action_add, action_delete))

    props_lines = ["Propositions:",
                   " ".join(propositions)]

    actions_lines = ["Actions:"] + \
                    [item for sublist in [get_action_lines(action) for action in actions] for item in
                     sublist]

    all_lines = props_lines + actions_lines

    write_to_file(all_lines, domain_file_name)


def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]

    init_on_props = [f"d_{n_ - 1}_on_p_0"] + [f"d_{i}_on_d_{i + 1}" for i in range(n_ - 1)]
    init_top_props = [f"d_0_top"] + [f"p_{i}_top" for i in range(1, m_)]
    init_smaller_than_props = [f"d_{s_i}_st_d_{b_i}" for b_i in range(n_) for s_i in range(b_i)] + \
                              [f"{disk}_st_{peg}" for disk in disks for peg in pegs]
    init_props = init_on_props + init_top_props + init_smaller_than_props

    goal_on_props = [f"d_{n_ - 1}_on_p_{m_ - 1}"] + [f"d_{i}_on_d_{i + 1}" for i in range(n_ - 1)]
    goal_props = goal_on_props

    problem_lines = [f"Initial state: {' '.join(init_props)}",
                     f"Goal state: {' '.join(goal_props)}"]

    write_to_file(problem_lines, problem_file_name)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
