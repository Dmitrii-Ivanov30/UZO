import matplotlib.pyplot as plt

def show_plt():
    manager = plt.get_current_fig_manager()
    manager.resize(1200, 800)
    plt.show()