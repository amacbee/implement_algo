# -*- coding: utf-8 -*-

from src.clustering.kmeans import KMeans
import random
import wx


class KMeansSimulator(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, 'DEMO - KMeans Clustering')
        self.panel = wx.Panel(self, size=(600, 600))
        self.panel.SetBackgroundColour('#ffffff')
        self.Fit()

        self.panel.Bind(wx.EVT_PAINT, self.paint)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.left_down)
        self.panel.Bind(wx.EVT_RIGHT_DOWN, self.right_down)

        # 5クラスタに分類
        # クラスタの色: red, blue, green, purple, navy
        self.n_clusters = 5
        self.cluster_colors = ('#e74c3c', '#2980b9', '#27ae60', '#8e44ad', '#34495e')

        # データの初期化
        self.cluster_centers = None
        self.X = None
        self.labels = None

        self.init_data()

        # kmeansの初期化
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def init_data(self):
        self.X = []
        self.labels = []
        for i in range(300):
            self.X.append((random.randint(0, 600), random.randint(0, 600)))
            self.labels.append(random.randint(0, self.n_clusters - 1))

    def paint(self, event):
        self.dc = wx.PaintDC(self.panel)

        # データ点を描画する
        for label, x in zip(self.labels, self.X):
            self.dc.SetPen(wx.Pen(self.cluster_colors[label], 5))
            self.dc.DrawPoint(x[0], x[1])

        # クラスタの重心を描画する
        self.cluster_centers = self.kmeans.cal_cluster_centers(self.X, self.labels)
        self.draw_cluster()

    def left_down(self, evt):
        self.update_cluster()
        self.Refresh()

    def right_down(self, evt):
        self.init_data()
        self.Refresh()

    def draw_cluster(self):
        for idx, center in enumerate(self.cluster_centers):
            self.dc.SetPen(wx.Pen(self.cluster_colors[idx], 2))
            c1, c2 = center
            self.dc.DrawLine(c1 - 5, c2 - 5, c1 + 5, c2 + 5)
            self.dc.DrawLine(c1 + 5, c2 - 5, c1 - 5, c2 + 5)

            self.dc.SetPen(wx.Pen(self.cluster_colors[idx], 1))
            for label, x in zip(self.labels, self.X):
                if label == idx:
                    self.dc.DrawLine(c1, c2, x[0], x[1])

    def update_cluster(self):
        for idx, x in enumerate(self.X):
            self.labels[idx] = self.kmeans.cal_nearest_cluster(x, self.cluster_centers)


if __name__ == '__main__':
    app = wx.App()

    simulator = KMeansSimulator()
    simulator.Show()

    app.MainLoop()
