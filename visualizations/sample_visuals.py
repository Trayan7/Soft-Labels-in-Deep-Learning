import matplotlib.pyplot as plt
import numpy as np
from data_handling.datasets import (load_cifar10h_split_dataset, load_datagen,
                                    load_plankton_split)
from sklearn.utils import shuffle


def sample_preview():
    '''
    Generates and saves PDFs for random samples
    with images and class membership distributions.
    Also determines samples per class and fuzzyness.
    '''
    # Number of samples to generate PDF for.
    preview_count = 1000
    # Get CIFAR-10H preview
    train_x, train_y, test_x, test_y = load_cifar10h_split_dataset(1000)
    samples = np.concatenate([train_x, test_x])
    labels = np.concatenate([train_y, test_y])

    # Get Plankton preview
    #train_x, train_y, test_x, test_y = load_plankton(1000)
    #samples = np.concatenate([train_x, test_x])
    #labels = np.concatenate([train_y, test_y])

    # Shuffle samples to get random samples for visualization.
    samples, labels = shuffle(samples, labels)

    # Count samples per class.
    class_count = np.zeros(10, dtype=int)
    for l in labels:
        class_count[np.argmax(l)] += 1
    print(class_count)


    # Count certain and uncertain samples.
    threshold = 0.9
    hard = 0 # 1 for class
    soft = 0 # (1,threshold] for class
    fuzzy = 0 # under threshold

    for l in labels:
        max = np.amax(l)
        if max == 1.0:
            hard += 1
        elif max >= threshold:
            soft += 1
        else:
            fuzzy += 1
    hardproc = hard * 100 / len(labels)
    softproc = soft * 100 / len(labels)
    fuzzyproc = fuzzy * 100 / len(labels)

    print('certain labels: ' + str(hard) + ', corresponding to ' + str(hardproc) + '%')
    print('up to threshold: ' + str(soft) + ', corresponding to ' + str(softproc) + '%')
    print('under threshold: ' + str(fuzzy) + ', corresponding to ' + str(fuzzyproc) + '%')

    # Generate PDFs of samples with soft labels
    for i in range(preview_count):
        fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [5, 1]})
        # CIFAR10 classes
        #classes = ['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']
        classes = ['1','2','3','4','5','6','7','8','9','10']
        label = labels[i]
        axs[1].bar(classes,label)
        # Rotate class labels
        #for tick in axs[1].get_xticklabels():
        #    tick.set_rotation(90)
        axs[0].imshow(samples[i], cmap='Greys_r') # Remove cmap for RGB
        fig.savefig('./preview/PlanktonPreview' + str(i) + '.pdf')


