





import seaborn as sns
import matplotlib.pyplot as plt

figure, axes = [], []


def init():
    global figure, axes
    plt.ion()
    plt.show()
    sns.set_theme()
    figure, axes = plt.subplots(1)

def draw(world):
    global figure, axes

    axes.clear()

    pos_mat = world.get_positions_array()
    print(pos_mat)

    sns.scatterplot(
        x=pos_mat[:,0],
        y=pos_mat[:,1]
    )

    figure.canvas.draw_idle()
    figure.canvas.start_event_loop(0.01)
    figure.canvas.set_window_title("Interstel Multi-Agent Coordination (MAC) Demonstration")
