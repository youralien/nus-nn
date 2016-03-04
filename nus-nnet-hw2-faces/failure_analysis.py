import matplotlib.pyplot as plt
import numpy as np
import imvis

def investigate_mlp(X, y_true, y_pred):
    successes = np.where(y_true == y_pred)
    if len(successes) == 2:
        successes = successes[0] # take out columns, we only want rows
    failures = np.where(y_true != y_pred)[0]
    if len(failures) == 2:
        failures = failures[0]  # take out columns, we only want rows

    plot16(X, failures, 'failures/16failures.png')
    plot16(X, successes, 'failures/16successes.png')

def plot16(X, slicer, fig_outfile):
    plt.clf()
    Xslice = X[slicer][:16]
    for count in range(Xslice.shape[0]):
        axes_idx = count + 1
        plt.subplot(4, 4, axes_idx)
        im = imvis.make2D(Xslice[count])
        plt.imshow(im, cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(fig_outfile, bbox_inches='tight')