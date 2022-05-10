import sys


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
                action_pre = [f"{disk}_top", f"{dest}_top", f"{disk}_st_{dest}"]
                action_add = [f"{disk}_on_{dest}", f"{src}_top"]
                action_delete = [f"{dest}_top", f"{disk}_on_{src}"]
                actions.append((action_name, action_pre, action_add, action_delete))

    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file

    domain_file.close()


def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file
    "*** YOUR CODE HERE ***"

    problem_file.close()


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
