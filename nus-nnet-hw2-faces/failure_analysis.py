import matplotlib.pyplot as plt
import numpy as np
import imvis

def investigate_mlp(X, y_true, y_pred):
    failures = np.where(y_true != y_pred)
    failures = failures[0] # take out columns, we only want rows

    plt.clf()
    # plot first 16 failures
    Xfail = X[failures]
    Xfail = Xfail[:16]
    for count in range(Xfail.shape[0]):
        axes_idx = count + 1
        plt.subplot(4, 4, axes_idx)
        im = imvis.make2D(X[count])
        plt.imshow(im, cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig('failures/allfailures.png', bbox_inches='tight')