# -*- coding: utf-8 -*-

import random


class KMeans:
    def __init__(self, n_clusters, random_state=71, max_iter=1000):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter

    def euclid(self, v1, v2):
        """
        ベクトル v1 と v2 のユークリッド距離を返す
        """
        return sum([(vv1 - vv2) ** 2 for vv1, vv2 in zip(v1, v2)]) ** 0.5

    def cal_nearest_cluster(self, x, cluster_centers):
        """
        ベクトル x と各クラスタの中心ベクトル cluster_centers の距離を計算し
        最も近距離のクラスタのラベルを返す
        """
        results = [self.euclid(x, center) for center in cluster_centers]
        return results.index(min(results))

    def cal_cluster_centers(self, X, labels):
        """
        各クラスタの中心ベクトルを計算して返す
        """
        cluster_centers = [[0 for _ in range(len(X[1]))] for _ in range(self.n_clusters)]
        for label, x in zip(labels, X):
            cluster_centers[label] = [c + xx for c, xx in zip(cluster_centers[label], x)]
        for label, cluster_center in enumerate(cluster_centers):
            cluster_centers[label] = [center / labels.count(label) for center in cluster_center]
        return cluster_centers

    def run(self, X, init_labels=None):
        """
        KMeansの処理を実行する
        """
        # 1. 各要素にクラスタ番号をふる
        if init_labels:
            labels = init_labels[:]
        else:
            # init_labelsで与えられていない場合はランダムにクラスタ番号をふる
            labels = [random.randint(0, self.n_clusters - 1) for _ in range(n_samples)]
        old_labels = labels[:]

        for _ in range(self.max_iter):
            # 2. 各クラスタの重心を計算する
            cluster_centers = self.cal_cluster_centers(X, labels)

            # 3. 各要素を一番近いクラスタの重心に振り分け
            for idx, x in enumerate(X):
                labels[idx] = self.cal_nearest_cluster(x, cluster_centers)

            # 4. クラスタのラベルの割当に変更がなければ処理を終了，変更があれば 2 - 3を繰り返す
            if labels == old_labels:
                break
            else:
                old_labels = labels[:]
        return cluster_centers, labels
