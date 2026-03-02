import numpy as np
import matplotlib.pyplot as plt 
import pickle

def show_images(X, y=None, class_names=None, num_images=9):
    """
    Displays random images in a grid.
    
    Parameters:
    - X: numpy array of shape (N, H, W, C)
    - y: labels (optional)
    - class_names: list of class names (optional)
    - num_images: how many images to show
    """
    
    N = X.shape[0]
    indices = np.random.choice(N, num_images, replace=False)
    
    grid_size = int(np.sqrt(num_images))
    
    plt.figure(figsize=(8, 8))
    
    for i, idx in enumerate(indices):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(X[idx].astype(np.uint8))
        plt.axis("off")
        
        if y is not None:
            if class_names is not None:
                plt.title(class_names[y[idx]])
            else:
                plt.title(str(y[idx]))
    
    plt.tight_layout()
    plt.show()



def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        
        X = datadict[b'data']
        y = datadict[b'labels']
        
        X = X.reshape(-1, 3, 32, 32)
        X = X.transpose(0, 2, 3, 1)  # to (N, 32, 32, 3)
        
        return X, np.array(y)

def main():
    X, y = load_cifar_batch("datasets/cifar-10-batches-py/data_batch_1")
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    show_images(X, y, class_names, num_images=9)

if __name__ == "__main__":
    main()
