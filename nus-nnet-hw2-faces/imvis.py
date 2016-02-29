import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def make2D(vector, size=(101, 101)):
    print vector.shape
    return vector.reshape(size).T

def save_example_face(vector, fig_outfile):
    im = make2D(vector)
    plt.imshow(im, cmap='gray')
    plt.tight_layout()
    plt.savefig(fig_outfile, bbox_inches='tight')

if __name__ == "__main__":
    from faces import faces
    trX, trY, teX, teY = faces(contrast=True)
    save_example_face(teX[0], 'example_face_contrast.png')
