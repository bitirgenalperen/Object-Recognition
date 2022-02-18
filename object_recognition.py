import cv2
from datetime import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
import pickle
import sys
import os
import matplotlib.pyplot as plt

BEST_KMEANS, BEST_KNN = 768, 32


sift_parameters = [(0, 3, 0.04, 10, 1.6), (0, 3, 0.04, 10, 1.5),
                   (0, 3, 0.04, 12, 1.6), (0, 3, 0.045, 10, 1.6),
                   (0, 4, 0.04, 10, 1.6), (16, 3, 0.04, 10, 1.6)]
cur_n, cur_oct, cur_c, cur_e, sig = (0, 4, 0.045, 10, 1.6)

dense_parameters = [(4, 8, 8), (8, 4, 8), (8, 8, 4)]
bound, step_size, scale = (4, 4, 4)


class SpatialInstance:
    def __init__(self, name, hist):
        self.name = name
        self.hist = hist


# Gets the class names (assuming the test data will also contain the same classes)
class_names = os.listdir("the2_data/train/")


def dsift(img):
    keypoints = []
    w, h = img.shape
    for i in range(bound, w - bound + 1, step_size):
        for j in range(bound, h - bound + 1, step_size):
            keypoints.append(cv2.KeyPoint(float(i), float(j), scale))
    dsift = cv2.SIFT_create()
    return dsift.compute(img, keypoints)


def sift(img):
    s = cv2.SIFT_create(cur_n, cur_oct, cur_c, cur_e, sig)
    return s.detectAndCompute(img, None)


class SiftClassifier:

    def __init__(self, class_name, desc):
        self.class_name = class_name
        self.desc = desc

    # I have written this function to observe the results
    # in case of assignment or data-type problems
    def __repr__(self):
        return "-------------------" + '\n' + "Class: " + str(self.class_name) + '\n' + "Descriptor Shape: " + str(len(self.desc)) + '\n' + "-------------------"

    def dictionary_constructor(sift_type, cluster_count):
        start = dt.now()
        global class_names
        # data will contain a list of SiftClassifier objects
        data = []
        instance = []
        descriptors = np.ndarray(shape=(0, 128))
        for cl in class_names:
            cur_dir = "the2_data/train/" + cl + "/"
            cur_imgs = os.listdir(cur_dir)
            for idx, l in enumerate(cur_imgs):
                # using 0, image internally converted to gray-scale
                cur = cur_dir + l
                cur_img = cv2.imread(cur, 0)
                if(sift_type == "sift"):
                    kp, des = sift(cur_img)

                    if(len(kp) == 0):
                        des = np.zeros((1, 128), dtype=np.double)
                    data.append(SiftClassifier(cl, des.astype(np.double)))
                    descriptors = np.concatenate((descriptors, des))
                elif(sift_type == "dense"):
                    kp, des = dsift(cur_img)
                    data.append(SiftClassifier(cl, des.astype(np.double)))
                    descriptors = np.concatenate((descriptors, des))
                else:
                    print("TypeError: -- Unrecognized type: '" +
                          str(sift_type) + "' is not a valid type")
                    return

        # K-means clustering
        clusters = MiniBatchKMeans(
            n_clusters=cluster_count, batch_size=3072, random_state=0).fit(descriptors)

        pickle.dump(clusters, open("clusters" + str(cluster_count) +
                                   "_" + str(sift_type) + ".pkl", "wb"))

        # Calculate histogram of features
        for i in range(len(data)):
            features = np.zeros(cluster_count, dtype=np.double)
            words = clusters.predict(data[i].desc)

            for word in words:
                features[word] += 1

            # Normalize histogram and create instance obj
            instance.append(SpatialInstance(
                data[i].class_name, features/np.sum(features)))
        print("desc: ", descriptors.shape)

        pickle.dump(instance, open("train" + str(cluster_count) +
                                   "_" + str(sift_type) + ".pkl", "wb"))
        print("Time Taken: ", dt.now() - start)

    def query_constructor(rel_dir, sift_type, cluster_count):
        query = []
        instance = []
        for cl in class_names:
            cur_dir = "the2_data/" + rel_dir + "/" + cl + "/"
            cur_imgs = os.listdir(cur_dir)
            for idx, l in enumerate(cur_imgs):
                # using 0, image internally converted to gray-scale
                cur = cur_dir + l
                cur_img = cv2.imread(cur, 0)
                if(sift_type == "sift"):
                    kp, des = sift(cur_img)

                    if(len(kp) == 0):
                        des = np.zeros((1, 128), dtype=np.double)
                    query.append(SiftClassifier(cl, des.astype(np.double)))
                elif(sift_type == "dense"):
                    kp, des = dsift(cur_img)
                    query.append(SiftClassifier(cl, des.astype(np.double)))
                else:
                    print("TypeError: -- Unrecognized type: '" +
                          str(sift_type) + "' is not a valid type")
                    return

        clusters = pickle.load(open("clusters" + str(cluster_count) +
                               "_" + str(sift_type) + ".pkl", "rb"))

        for i in range(len(query)):
            features = np.zeros(cluster_count, dtype=np.double)
            words = clusters.predict(query[i].desc)

            for word in words:
                features[word] += 1

            # Normalize
            instance.append(SpatialInstance(
                query[i].class_name, features/np.sum(features)))

        pickle.dump(instance, open(rel_dir + str(cluster_count) +
                                   "_" + str(sift_type) + ".pkl", "wb"))


def calculate_distances(train_hist, hist):
    return np.sum(((train_hist-hist)*(train_hist-hist)), axis=1)**(0.5)


def majority_voting(distances, labels, k):
    t = np.concatenate(
        (distances.reshape(-1, 1), labels.reshape(-1, 1)), axis=1)
    sortedArr = t[np.argsort(t[:, 0])]
    u, p = np.unique(sortedArr[:k, 1], return_inverse=True)
    return u[np.bincount(p).argmax()]


def knn(eval_type, sift_type, cluster_count, neigboor_count, reveal_misclassified=False):
    train_data = pickle.load(open("train" + str(cluster_count) +
                             "_" + str(sift_type) + ".pkl", 'rb'))
    datum, label = np.asarray([elem.hist for elem in train_data]), np.asarray(
        [elem.name for elem in train_data])
    if(eval_type == "validation"):
        eval_data = pickle.load(open(eval_type + str(cluster_count) +
                                "_" + str(sift_type) + ".pkl", 'rb'))
        num_correct, total_score = 0.0, len(eval_data)
        a, p = [], []
        for index, test_elem in enumerate(eval_data):
            actual_label = eval_data[index].name
            predicted_label = majority_voting(calculate_distances(
                datum, test_elem.hist), label, neigboor_count)
            a.append(actual_label)
            p.append(predicted_label)
            if (actual_label == predicted_label):
                num_correct += 1
        if(reveal_misclassified == False):
            return (num_correct/total_score)
        elif(reveal_misclassified == True):
            print(num_correct/total_score)
            k = confusion_matrix(a, p)
            cm = pd.DataFrame(k, index=class_names, columns=class_names)
            plt.figure(figsize=(16, 10))
            t = sns.heatmap(cm, annot=True, cmap="YlGnBu",
                            linewidths=1.0, xticklabels=True, yticklabels=True)
            t.set_title("Confusion Matrix using the Best Configuration")
            t.set_xlabel("Predicted Class-names")
            t.set_ylabel("Actual Class-names")
            t.get_figure().savefig("confussion_matrix.png")
    elif(eval_type == "test"):
        clusters = pickle.load(
            open("clusters" + str(BEST_KMEANS) + "_" + str("dense") + ".pkl", "rb"))
        query = []
        instance = []
        cur_dir = "the2_data/test/"
        cur_imgs = os.listdir(cur_dir)
        for idx, l in enumerate(cur_imgs):
            cur = cur_dir + l
            cur_img = cv2.imread(cur, 0)
            kp, des = dsift(cur_img)
            query.append(SiftClassifier(str(l), des.astype(np.double)))
        for i in range(len(query)):
            features = np.zeros(cluster_count, dtype=np.double)
            words = clusters.predict(query[i].desc)

            for word in words:
                features[word] += 1

            # Normalize
            instance.append(SpatialInstance(
                query[i].class_name, features/np.sum(features)))
        for index, test_elem in enumerate(instance):
            predicted_label = majority_voting(calculate_distances(
                datum, test_elem.hist), label, neigboor_count)
            h = str(test_elem.name) + ": " + str(predicted_label) + "\n"
            with open("test_predictions.txt", 'a') as f:
                f.write(h)


def custom_process_selector(args):
    if(args[0] == "train" and len(args) == 3):
        print("train")
        SiftClassifier.dictionary_constructor(args[1], int(args[2]))

    elif(args[0] == "validation" and len(args) == 3):
        print("validate")
        SiftClassifier.query_constructor(args[0], args[1], int(args[2]))

    elif (args[0] == "evaluate" and len(args) == 5):
        print(knn(args[1], args[2], int(args[3]), int(args[4])))

    elif (args[0] == "run" and len(args) == 5):
        # (eval_dir, sift_type, cluster_count, neigboor_count)
        SiftClassifier.dictionary_constructor(args[2], int(args[3]))
        SiftClassifier.query_constructor(args[1], args[2], int(args[3]))
        print(knn(args[1], args[2], int(args[3]), int(args[4])))

    elif (args[0] == "run-best"):
        # (eval_dir, sift_type, cluster_count, neigboor_count)
        SiftClassifier.dictionary_constructor("dense", BEST_KMEANS)
        SiftClassifier.query_constructor("validation", "dense", BEST_KMEANS)
        knn("validation", "dense", BEST_KMEANS,
            BEST_KNN, reveal_misclassified=True)

    elif(args[0] == "run-test"):
        SiftClassifier.dictionary_constructor("dense", BEST_KMEANS)
        knn("test", "dense", BEST_KMEANS, BEST_KNN)

    elif(args[0] == "part1-sift" and len(args) == 1):
        for elem in sift_parameters:
            global cur_n, cur_oct, cur_c, cur_e, sig
            cur_n, cur_oct, cur_c, cur_e, sig = elem
            SiftClassifier.dictionary_constructor("sift", 128)
            SiftClassifier.query_constructor("validation", "sift", 128)
            acc = knn("validation", "sift", 128, 8)
            print("-------------------------------------------------------------")
            print("nfeatures: ", cur_n, " nOctave: ", cur_oct,
                  " contrastThres: ", cur_c, " edgeThres: ", cur_e, " sigma: ", sig)
            print("acc: ", acc)
            print("-------------------------------------------------------------")
            with open("part1_sift.txt", 'a') as f:
                f.write(
                    "-------------------------------------------------------------\n")
                f.write("nfeatures: " + str(cur_n) + " nOctave: " + str(cur_oct) + " contrastThres: " +
                        str(cur_c) + " edgeThres: " + str(cur_e) + " sigma: " + str(sig) + " \n")
                f.write("acc: " + str(acc) + " \n")
                f.write(
                    "-------------------------------------------------------------\n")
    elif(args[0] == "part1-dense" and len(args) == 1):
        for elem in dense_parameters:
            global bound, step_size, scale
            bound, step_size, scale = elem
            SiftClassifier.dictionary_constructor("dense", 128)
            SiftClassifier.query_constructor("validation", "dense", 128)
            acc = knn("validation", "dense", 128, 8)
            print("-------------------------------------------------------------")
            print("bound: " + str(bound) + " step_size: " +
                  str(step_size) + " scale: " + str(scale))
            print("acc: ", acc)
            print("-------------------------------------------------------------")
            with open("part1_dense.txt", 'a') as f:
                f.write(
                    "-------------------------------------------------------------\n")
                f.write("bound: " + str(bound) + " step_size: " +
                        str(step_size) + " scale: " + str(scale) + " \n")
                f.write(
                    "acc: " + str(acc) + " \n")
                f.write(
                    "-------------------------------------------------------------\n")
    elif(args[0] == "part2" and len(args) == 1):
        for cl in [768, 1024]:
            SiftClassifier.dictionary_constructor("dense", cl)
            SiftClassifier.query_constructor("validation", "dense", cl)
            print("**********************")
            print("Feature extractor: " + "dense" +
                  " :: Cluster Count: " + str(cl))
            print(str(8) + "-NN")
            print("**********************")
            print("Accuracy: ", knn("validation", "dense", cl, 8))
            print("----------------------")

    elif(args[0] == "part3" and len(args) == 1):
        for n in [8, 16, 32, 64, 24]:
            SiftClassifier.dictionary_constructor("dense", BEST_KMEANS)
            SiftClassifier.query_constructor(
                "validation", "dense", BEST_KMEANS)
            print("**********************")
            print("Feature extractor: " + "dense" +
                  " :: Cluster Count: " + str(BEST_KMEANS))
            print(str(n) + "-NN")
            print("**********************")
            print("Accuracy: ", knn("validation", "dense", BEST_KMEANS, n))
            print("----------------------")

    # clears the saved data of .pkl files
    elif (args[0] == "clear" and len(args) == 1):
        items = os.listdir(".")
        for item in items:
            if item.endswith(".pkl"):
                os.remove(os.path.join(".", item))
        print("clear")

    else:
        print("CANNOT FIND ANY RECOGNIZABLE PARAMETER")


if __name__ == '__main__':
    if(len(sys.argv) >= 2):
        custom_process_selector(sys.argv[1:])
    else:
        print("CANNOT FIND ANY GIVEN PARAMETER")
