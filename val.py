import torch
import numpy as np
import matplotlib.pyplot as plt


def compare_models(gat_logfile, cnn_logfile):
    f1 = open(gat_logfile, mode='r')
    f2 = open(cnn_logfile, mode='r')
    str_list1 = f1.readlines()
    f1.close()
    str_list2 = f2.readlines()
    f2.close()
    train_gat_list1 = []
    train_cnn_list1 = []
    test_gat_list1 = []
    test_cnn_list1 = []

    for idx in range(1, 51):
        train_loss, test_loss = str_list1[idx].split(',')
        train_gat_list1.append(float(train_loss))
        train_cnn_list1.append(float(str_list2[idx].split(',')[0]))
        test_gat_list1.append(float(test_loss[:-2]))
        test_cnn_list1.append(float(str_list2[idx].split(',')[1][:-2]))

    train_gat_list2 = []
    train_cnn_list2 = []
    test_gat_list2 = []
    test_cnn_list2 = []
    for idx in range(52, 102):
        train_loss, test_loss = str_list1[idx].split(',')
        train_gat_list2.append(float(train_loss))
        train_cnn_list2.append(float(str_list2[idx].split(',')[0]))
        test_gat_list2.append(float(test_loss[:-2]))
        test_cnn_list2.append(float(str_list2[idx].split(',')[1][:-2]))

    train_gat_list3 = []
    train_cnn_list3 = []
    test_gat_list3 = []
    test_cnn_list3 = []
    for idx in range(103, 153):
        train_loss, test_loss = str_list1[idx].split(',')
        train_gat_list3.append(float(train_loss))
        train_cnn_list3.append(float(str_list2[idx].split(',')[0]))
        test_gat_list3.append(float(test_loss[:-2]))
        test_cnn_list3.append(float(str_list2[idx].split(',')[1][:-2]))

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
    # ax1.plot(range(1, 51), test_gat_list1, "ro-", label="GAT")
    # ax1.plot(range(1, 51), test_cnn_list1, "b*-", label="CNN")
    # ax1.set_title("test dataset1")
    # ax1.legend()

    # ax2.plot(range(1, 51), test_gat_list2, "ro-", label="GAT")
    # ax2.plot(range(1, 51), test_cnn_list2, "b*-", label="CNN")
    # ax2.set_title("test dataset2")
    # ax2.legend()

    # ax3.plot(range(1, 51), test_gat_list3, "ro-", label="GAT")
    # ax3.plot(range(1, 51), test_cnn_list3, "b*-", label="CNN")
    # ax3.set_title("test dataset3")
    # ax3.legend()



    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    # ax1.plot(range(1, 51), train_gat_list1, "ro-", label="GAT")
    # ax1.plot(range(1, 51), train_cnn_list1, "b*-", label="CNN")
    # ax1.set_title("(a)train dataset1")
    # ax1.legend()

    # ax2.plot(range(1, 51), test_gat_list1, "ro-", label="GAT")
    # ax2.plot(range(1, 51), test_cnn_list1, "b*-", label="CNN")
    # ax2.set_title("(b)test dataset2")
    # ax2.legend()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    # ax1.plot(range(1, 51), train_gat_list2, "ro-", label="GAT")
    # ax1.plot(range(1, 51), train_cnn_list2, "b*-", label="CNN")
    # ax1.set_title("(a)train dataset2")
    # ax1.legend()

    # ax2.plot(range(1, 51), test_gat_list2, "ro-", label="GAT")
    # ax2.plot(range(1, 51), test_cnn_list2, "b*-", label="CNN")
    # ax2.set_title("(b)test dataset2")
    # ax2.legend()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    # ax1.plot(range(1, 51), train_gat_list3, "ro-", label="GAT")
    # ax1.plot(range(1, 51), train_cnn_list3, "b*-", label="CNN")
    # ax1.set_title("(a)train dataset3")
    # ax1.legend()

    # ax2.plot(range(1, 51), test_gat_list3, "ro-", label="GAT")
    # ax2.plot(range(1, 51), test_cnn_list3, "b*-", label="CNN")
    # ax2.set_title("(b)test dataset3")
    # ax2.legend()



    plt.scatter(range(5, 51, 5), test_gat_list3[4:50:5], c="red", marker="o", label="GAT")
    plt.scatter(range(5, 51, 5), test_cnn_list3[4:50:5], c="blue", marker="*", label="CNN")
    plt.xlim((0,55))
    plt.title("test dataset3")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    compare_models("train_log.txt", "train_cnn_log.txt")

