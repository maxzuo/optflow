import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from cv2 import cvtColor, COLOR_RGB2BGR

from optflow.util import dist, tqdm

def display_kmeans(states:np.ndarray, labels:np.ndarray, centers:np.ndarray) -> np.ndarray:
    """
    Creates a display of the state of the centers for

    Uses sklearn's PCA and MinMaxScaler functionalities for
    projection onto 2D and scaling, respectively.

    Inputs:
    - states: a NxD array of the states to display
    - labels: a N length array of the labels for each state
    - centers: KxD array for state centers to display

    Outputs:
    - plot: an ndarray image in BGR color space displaying the plot

    """
    k = centers.shape[0]

    # bring down to 2 dimensions
    if k > 2:
        states = dist(states, centers)

        pca = PCA(n_components=2)
        states = pca.fit_transform(states)

        centers = dist(centers, centers)
        centers = pca.transform(centers)

    scaler = MinMaxScaler()

    scaler.fit(np.vstack([states, centers]))
    states = scaler.transform(states)
    centers = scaler.transform(centers)


    fig = plt.figure()
    r = list(range(k))
    w,h = fig.canvas.get_width_height()

    for i,flow in enumerate(tqdm(states)):

        fig.gca().axis('off')
        fig.gca().set_xlim(right=1.05)
        fig.gca().set_xlim(left=-0.05)
        fig.gca().set_ylim(top=1.05)
        fig.gca().set_ylim(bottom=-0.05)
        fig.tight_layout()

        fig.gca().scatter(centers.T[0], centers.T[1], 50, r)
        fig.gca().scatter((*(-np.ones(k)),flow[0],),(*(-np.ones(k)),flow[1],),500,[*r, labels[i]])
        fig.canvas.draw()
        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h,w,3))
        # plot = resize(plot, None, fx=0.5, fy=0.5)
        plot = cvtColor(plot, COLOR_RGB2BGR)
        fig.canvas.flush_events()
        fig.clear()
        yield plot