from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import l1_min as l1ls
import csv

# load image data
def load_data(class_number, sample_number, debug=False):
    dic = None
    for i in range(class_number):
        if os.path.exists('orl_faces/s' + str(i + 1)):
            for j in range(int(sample_number * 0.7)):
                if os.path.isfile('orl_faces/s' + str(i + 1) + '/' + str(j + 1) + '.pgm'):
                    img = io.imread('orl_faces/s' + str(i + 1) + '/' + str(j + 1) + '.pgm', as_grey=True)
                    if debug:
                        io.imshow(img)
                        io.show()
                    tmp = normalize(np.reshape(img, [1, -1]))

                    if dic is None:
                        dic = tmp
                    else:
                        dic = np.append(dic, tmp, axis=0)
        else:
            break
    return np.array(dic).T


def random_test_sample(class_number, sample_number, dist=False, overlay=False):

    class_num_random = random.randint(1, class_number)
    sample_num_random = random.randint(1, sample_number)

    print(class_num_random)
    print(sample_num_random)
    if os.path.isfile('orl_faces/s' + str(class_num_random) + '/' + str(sample_num_random) + '.pgm'):
        img = io.imread('orl_faces/s' + str(class_num_random) + '/' + str(sample_num_random) + '.pgm', as_grey=True)
        tmp = normalize(np.reshape(img, [1, -1]))
        tmp_n = tmp
        if dist:
            my_noise = noise(0, 0.003, train_data.shape[0])
            tmp_n = tmp + my_noise
        if overlay:
            class_overlay = random.randint(0, class_number)
            sample_overlay = random.randint(0, sample_number)
            n = io.imread('orl_faces/s' + str(class_overlay) + '/' + str(sample_overlay) + '.pgm', as_grey=True)
            n_n = normalize(np.reshape(n, [1, -1]))
            tmp_n = 0.6*tmp_n + 0.4*n_n
        return tmp.T, tmp_n.T
    else:
        print("Test is invalid!")
        return False


def random_test(class_number, sample_number, dic, dist=False, overlay=False):
    my_y_pure, my_y = random_test_sample(class_number, sample_number, dist, overlay)
    [my_x, my_status, my_hist] = l1ls.l1ls(dic, my_y, 0.01)
    residue, delta = residue_delta(dic, my_y, my_x, 40, int(10 * 0.7))
    my_label = valid_classification(residue, delta, my_x, 40, 0.85)

    pure_y = np.reshape(my_y_pure, [112, 92])
    noise_y = np.reshape(my_y, [112, 92])
    ground_truth = io.imread('orl_faces/s' + str(my_label) + '/1.pgm', as_grey=True)

    return pure_y, noise_y, ground_truth


def tot_test(class_number, sample_number):
    my_test = None
    for i in range(class_number):
        for j in range(int(sample_number * 0.7), sample_number):
            img = io.imread('orl_faces/s' + str(i + 1) + '/' + str(j + 1) + '.pgm', as_grey=True)
            tmp = normalize(np.reshape(img, [1, -1]))
            if my_test is None:
                my_test = tmp.T
            else:
                my_test = np.append(my_test, tmp.T, 1)
    return my_test


def normalize(sample):
    ret = sample / np.linalg.norm(sample)
    return ret


def noise(mu, sigma, length):
    s = np.random.normal(mu, sigma, length)
    return s

# calculate the residue
def residue_delta(trained, test_sample, raw_x, class_number, sample_number):
    residue = []
    delta = []
    for i in range(0, class_number):
        tmp = np.zeros(len(raw_x))
        for j in range(i * sample_number, i * sample_number + sample_number):
            tmp[j] = raw_x[j]
        delta.append(tmp)
        residue.append(np.linalg.norm(test_sample - trained.dot(tmp)))
    return residue, delta


def valid_classification(residue, delta, my_x, class_number, tau):
    max_delta = -1
    for c in range(class_number):
        tmp = np.linalg.norm(delta[c], ord=1)
        if tmp >= max_delta:
            max_delta = tmp
    norm_x = np.linalg.norm(my_x, ord=1)
    sci = ((class_number * max_delta / norm_x) - 1) / (class_number - 1)
    print(sci)
    if sci >= tau:
        return residue.index(min(residue)) + 1
    else:
        return residue.index(min(residue)) + 1


def accuracy(predicted):
    count = 0
    for row in range(predicted.shape[0]):
        if int(predicted[row, 0]) == row // 3 + 1:
            count += 1
    acc = count/predicted.shape[0]
    print(acc)


if __name__ == '__main__':
    # demo = True
    # trained_yet = True

    demo = False
    trained_yet = False

    train_data = load_data(40, 10)
    test_data = tot_test(40, 10)

    if demo:
        original = [0]*4
        noised = [0]*4
        truth = [0]*4
        original[0], noised[0], truth[0] = random_test(40, 10, train_data, False, False)
        original[1], noised[1], truth[1] = random_test(40, 10, train_data, True, False)
        original[2], noised[2], truth[2] = random_test(40, 10, train_data, False, True)
        original[3], noised[3], truth[3] = random_test(40, 10, train_data, True, True)
        fig = plt.figure()
        for i in range(4):
            ax1 = fig.add_subplot(4, 3, 3*i+1)
            ax2 = fig.add_subplot(4, 3, 3*i+2)
            ax3 = fig.add_subplot(4, 3, 3*i+3)
            ax1.imshow(original[i])
            ax2.imshow(noised[i])
            ax3.imshow(truth[i])
        plt.show()
    else:
        if not trained_yet:
            labels = []
            for i in range(test_data.shape[1]):
                y = test_data[:, i]
                [x, status, hist] = l1ls.l1ls(train_data, y, 0.01, quiet=True)
                my_residue, my_delta = residue_delta(train_data, y, x, 40, int(10 * 0.7))
                label = valid_classification(my_residue, my_delta, x, 40, 0.5)
                labels.append(label)
            labels = np.array(labels)
            np.savetxt('result.csv', np.array(labels, dtype='float'), delimiter=',')

        labels = []
        with open('result.csv') as f:
            rows = csv.reader(f, delimiter=',')
            for row in rows:
                labels.append(row)
        labels = np.array(labels, dtype=float)

        print(labels)
        accuracy(labels)
