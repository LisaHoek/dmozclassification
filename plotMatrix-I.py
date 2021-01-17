import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
  
def plot_confusion_matrix(data, labels, title, output_filename):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(20, 6))
 
    plt.title(title)
 
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='g')
 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation = 0)
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
 
# define data
matrix = [[1.6002e+04, 2.8530e+03, 2.2920e+03, 3.5400e+02, 3.0100e+02,
              3.8000e+02, 1.7700e+02, 2.1300e+02, 2.1300e+02, 1.5100e+02,
              2.5000e+01, 4.4000e+01, 9.0000e+00],
             [2.9250e+03, 1.5232e+04, 2.0880e+03, 3.3300e+02, 2.3300e+02,
              1.8700e+02, 1.8500e+02, 2.8000e+02, 1.6000e+02, 3.8000e+01,
              6.8000e+01, 2.7000e+01, 1.8000e+01],
             [2.6610e+03, 1.9900e+03, 1.2438e+04, 3.0900e+02, 4.2300e+02,
              3.5200e+02, 1.7000e+02, 7.5500e+02, 1.2900e+02, 1.4300e+02,
              2.2000e+01, 4.4000e+01, 7.0000e+00],
             [1.9150e+03, 1.7920e+03, 1.6020e+03, 3.7390e+03, 1.5400e+02,
              1.9700e+02, 1.8900e+02, 2.3700e+02, 4.3000e+01, 5.9000e+01,
              2.3000e+01, 3.5000e+01, 5.0000e+00],
             [1.1270e+03, 1.1020e+03, 2.2700e+03, 1.1800e+02, 4.5370e+03,
              2.7800e+02, 6.7000e+01, 1.0800e+02, 6.7000e+01, 2.9000e+01,
              4.5000e+01, 1.3000e+01, 3.0000e+00],
             [1.5490e+03, 8.7300e+02, 1.8330e+03, 1.8200e+02, 3.7600e+02,
              4.2370e+03, 6.4000e+01, 8.6000e+01, 2.7400e+02, 7.7000e+01,
              1.7000e+01, 2.7000e+01, 3.0000e+00],
             [1.4230e+03, 1.4550e+03, 1.1490e+03, 3.4400e+02, 6.8000e+01,
              5.4000e+01, 4.5440e+03, 9.0000e+01, 5.5000e+01, 2.4000e+01,
              1.8000e+01, 8.0000e+00, 8.0000e+00],
             [9.5000e+02, 1.4210e+03, 2.2890e+03, 2.9200e+02, 1.0100e+02,
              8.7000e+01, 1.2200e+02, 3.0770e+03, 2.4000e+01, 8.4000e+01,
              1.9000e+01, 4.1000e+01, 2.0000e+00],
             [1.2370e+03, 6.5900e+02, 1.1230e+03, 9.8000e+01, 1.3800e+02,
              2.8900e+02, 1.4900e+02, 7.5000e+01, 1.4930e+03, 5.2000e+01,
              1.3000e+01, 8.0000e+00, 5.0000e+00],
             [1.2360e+03, 4.9600e+02, 9.9000e+02, 8.6000e+01, 4.2000e+01,
              1.2300e+02, 6.9000e+01, 5.8000e+01, 6.3000e+01, 1.9150e+03,
              6.0000e+00, 1.6000e+01, 2.0000e+00],
             [4.4600e+02, 9.8100e+02, 3.8200e+02, 9.2000e+01, 2.0200e+02,
              4.8000e+01, 6.3000e+01, 4.2000e+01, 1.3000e+01, 9.0000e+00,
              1.2800e+03, 1.0000e+00, 0.0000e+00],
             [4.4700e+02, 4.1400e+02, 3.2800e+02, 1.1100e+02, 3.9000e+01,
              8.6000e+01, 2.5000e+01, 5.9000e+01, 1.4000e+01, 2.4000e+01,
              9.0000e+00, 6.8900e+02, 5.0000e+00],
             [2.6800e+02, 1.6800e+02, 1.6000e+02, 1.8000e+01, 1.7000e+01,
              2.0000e+01, 4.8000e+01, 5.0000e+00, 2.2000e+01, 6.0000e+00,
              3.0000e+00, 0.0000e+00, 1.3600e+02]]

matrixdf = pd.DataFrame(matrix)
normalize = matrixdf.astype('float') / matrixdf.sum(axis=1)[:, np.newaxis]

# define labels
labels = ["A", "B", "C", "D", "E", "F"]
classes = [0.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

dictLabels = {8.0: 'Reference', 11.0: 'Home', 9.0: 'Health', 7.0: 'Shopping', 6.0: 'Sports', 12.0: 'News', 3.0: 'Recreation', 4.0: 'Computers', 5.0: 'Science', 0.0: 'Society', 10.0: 'Games', 2.0: 'Business', 1.0: 'Arts'}

classesName = []
for c in sorted(classes):
	classesName.append(dictLabels.get(c))

print(classesName)
 
# create confusion matrix
plot_confusion_matrix(matrix, classesName, "Confusion Matrix", "confusion_matrix.png")
plot_confusion_matrix(normalize.round(2), classesName, "Normalised Confusion Matrix", "normalized_confusion_matrix.png")
