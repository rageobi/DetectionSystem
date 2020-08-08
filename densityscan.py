from typing import NamedTuple
import math
import numpy as np
import random
count = 0


class Cluster():
    def __init__(self, x=None, y=None):
        global count
        self.x = x
        self.y = y
        self.id = count = count + 1
        self.cluster = -2

    def __repr__(self):
        return [self.x], [self.y], [self.cluster]

    def CheckValidPoints(self, point, x_dist, y_dist) -> int:
        #same point as the base cluster
        if self.x == point.x and self.y == point.y:
            return 2
        #Within the mentioned distance from Base cluster
        elif self.GetDistance(point, 1) <= x_dist and self.GetDistance(point, 2) <= y_dist:
            return 1
        #Not Within the mentioned distance from Base cluster
        else:
            return 0

    def GetDistance(self, p2, check):
        #Get X Distance
        if check == 1:
            return round(abs(p2.x - self.x), 5)
        #Get Y Distance
        elif check == 2:
            return round(abs(p2.y - self.y), 5)
        #Wrong option
        else:
            return -1


class ClusterLists():
    #cluster_val = 0
    def __init__(self):
        self.cluster_list = []
        self.randoms = []
        self.cluster_val = 1

    def get_cluster_labels(self,verbose=False):
        st = []
        for x in self.cluster_list:
            st.append(x.cluster)
        if verbose:
            print(("     {} clusters for the frame is").format(len(st)))
        return st

    def update_random(self):
        if (type(self.cluster_list).__module__ != np.__name__):
            self.reshape()
        if (len(self.randoms) != len(self.cluster_list)):
            self.randoms = list(range(self.cluster_list.shape[0]))

    def cluster_cluster(self,x_dist,y_dist,verbose):
        self.update_random()
        for i in range(0, len(self.cluster_list)):#len(self.randoms)):
            #choice = random.choice(self.randoms)
            self.CheckValidClusters(self.cluster_list[i], x_dist, y_dist)
            #self.randoms.remove(choice)
        return np.array(self.get_cluster_labels(verbose))

    def append(self, cluster: Cluster):
        self.cluster_list.append(cluster)

    def reshape(self):
        self.cluster_list = np.array(self.cluster_list)  # .reshape(shape_0,)

    def CheckValidClusters(self, base_cluster, x_dist, y_dist):
        if base_cluster.cluster == -2:
            for cluster in self.cluster_list:
                if cluster.cluster == -2:
                    d_check = base_cluster.CheckValidPoints(
                        cluster, x_dist, y_dist)
                    if d_check == 1:
                        cluster.cluster = self.cluster_val
            base_cluster.cluster = self.cluster_val
            self.cluster_val += 1


def testMethod():
    p1 = Cluster(1, 2)
    p2 = Cluster(2, 3)

    p = ClusterLists()
    p.append(p1)
    p.append(p2)
    p.append(Cluster(3, 1))
    p.append(Cluster(1, 1))
    p.append(Cluster(2, 2))
    p.append(Cluster(3, 3))
    p.append(Cluster(1, 3))
    p.append(Cluster(2, 1))
    p.append(Cluster(3, 2))
    p.append(Cluster(4, 1))
    p.append(Cluster(2, 4))
    p.append(Cluster(4, 4))
    p.append(Cluster(3, 4))
    p.append(Cluster(2, 4))

    p.update_random()
    print(p.randoms)
    for i in range(0, len(p.randoms)):
        choice = random.choice(p.randoms)
        p.CheckValidClusters(p.cluster_list[choice], 1, 1)
        p.randoms.remove(choice)
        print(p.randoms)

    for cluster in p.cluster_list:
        print("x ={}, y={}, cluster{}".format(
            cluster.x, cluster.y, cluster.cluster))
    s = p.cluster_list[0].CheckValidPoints(p.cluster_list[1], 1.5, 1.5)

    p.reshape()
    p3 = p1.CheckValidPoints(p2, 1.5, 1.5)
    print(p1.z)

