import numpy as np


# class Poisson_disc:
#     def __init__(self,x, y,r):
#         self.x = x
#         self.y = y
#         self.sample = []  # [(px,py)]
#         self.activelist = []
#         self.r = r
#         self.cell = {}  # {(cs,cy):pidx}
#         self.cell_x = r / np.sqrt(2)
#         self.cell_y = r / np.sqrt(2)
#         self.cell_x_num = int(self.x / self.cell_x)
#         if self.x - self.cell_x_num*self.cell_x > 0:
#             self.cell_x_num = self.cell_x_num + 1
#         self.cell_y_num = int(self.y / self.cell_y)
#         if self.y - self.cell_y_num * self.cell_y > 0:
#             self.cell_y_num = self.cell_y_num + 1
#         # init cell
#         for cx in range(self.cell_x_num):
#             for cy in range(self.cell_y_num):
#                 self.cell[(cx,cy)] = None

#         # plt.xticks(np.arange(self.x))
#         # plt.yticks(np.arange(self.y))
#     def get_cell(self,px,py):
#         return int(px/self.cell_y), int(py/self.cell_x)
#     def get_neighbor_cell(self,icx,icy):
#         dxdy = [(-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
#                 (-2, 0), (-1, 0), (1, 0), (2, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
#                 (-1, 2), (0, 2), (1, 2), (0, 0)]
#         neighbours = []
#         for cx, cy in dxdy:
#             nx, ny = icx + cx, icy + cy
#             if not (0 <= nx < self.cell_x_num and 0 <= ny < self.cell_y_num):
#                 continue
#             elif self.cell[(nx, ny)] is not None:
#                 neighbours.append(self.cell[(nx, ny)])
#         return neighbours
#     def point_valid(self,px, py):
#         p_cell_x, p_cell_y = self.get_cell(px,py)
#         neighbors = self.get_neighbor_cell(p_cell_x,p_cell_y)
#         for pidx in neighbors:
#             n_x, n_y = self.sample[pidx]
#             distance2 = (n_x - px) ** 2 + (n_y - py) ** 2
#             if distance2 <= self.r ** 2:
#                 # The points are too close, so pt is not a candidate.
#                 return False
#         return True
#     def get_point(self,k,px,py):
#         r = self.r
#         for i in range(k):
#             rho, theta = np.random.uniform(r, 2 * r), np.random.uniform(0, 2 * np.pi)
#             n_px, n_py = np.round(px + rho * np.cos(theta)), np.round(py + rho * np.sin(theta))
#             if not (r**2 < (n_px-px)**2 + (n_py - py)**2 < 4*r**2 and 0 <= n_px < self.y and 0 <= n_py < self.x):
#                 continue
#             if self.point_valid(n_px,n_py):
#                 return n_px,n_py
#         return False
#     def run(self):
#         init_px, init_py = self.y//2,self.x//2
#         self.sample = [(init_px, init_py),]
#         self.activelist.append(0)
#         p_c_x, p_c_y = self.get_cell(init_px, init_py)
#         self.cell[(p_c_x, p_c_y)] = 0
#         idx = 1
#         while self.activelist:
#             samples = self.sample
#             # plt.scatter(*zip(*samples), s=6, color='k', alpha=0.5, lw=0)
#             pidx = np.random.choice(self.activelist)
#             px, py = self.sample[pidx]
#             # plt.scatter(px, py, s=6, color='g', alpha=1, lw=0)
#             p = self.get_point(30, px, py)
#             if p:
#                 self.sample.append((p[0], p[1]))
#                 self.activelist.append(idx)
#                 p_c_x, p_c_y = self.get_cell(p[0],p[1])
#                 self.cell[(p_c_x, p_c_y)] = idx
#                 # plt.scatter(p[0], p[1], s=6, color='r', alpha=1, lw=0)
#                 idx = idx + 1
#             else:
#                 self.activelist.remove(pidx)
#             # print('idx',pidx)


#             # plt.xlim(0, self.x)
#             # plt.ylim(0, self.y)
#             # plt.axis('on')
#             # plt.pause(0.1)
#             # plt.cla()
#         print('expectation:{}',len(self.sample)/(self.x*self.y))
#         # plt.cla()
#         return self.sample

class VD_Poisson_disc:
    def __init__(self, x, y, alpha, radius, k):
        # find the center point (x/2, y/2)
        x_center, y_center = (x - 1) / 2, (y - 1) / 2
        self.x_centre, self.y_center = x_center, y_center
        self.alpha = alpha
        max_dist2 = x_center ** 2 + y_center ** 2
        # inital radius >= 1.5
        # r = max(np.sqrt(max_dist2) * alpha, 1.5)
        r = radius
        # super(VD_Poisson_disc,self).__init__(x,y,r)
        self.x = x
        self.y = y
        self.sample = []  # [(px,py)]
        self.activelist = []
        self.r = r
        self.k = k
        # split a rectange x, y to small cells
        self.cell = {}  # {(cs,cy):pidx}
        self.cell_x = r / np.sqrt(2)
        self.cell_y = r / np.sqrt(2)
        self.cell_x_num = int(np.ceil(self.x / self.cell_x))
        # if self.x - self.cell_x_num * self.cell_x > 0:
        #     self.cell_x_num = self.cell_x_num + 1
        self.cell_y_num = int(np.ceil(self.y / self.cell_y))
        # if self.y - self.cell_y_num * self.cell_y > 0:
        #     self.cell_y_num = self.cell_y_num + 1
        # init cell
        for cx in range(self.cell_x_num):
            for cy in range(self.cell_y_num):
                self.cell[(cx, cy)] = []

    def get_cell(self, px, py):
        # find the id of the cell by point (px, py)
        return int(px / self.cell_y), int(py / self.cell_x)

    def point_valid(self, px, py):
        p_cell_x, p_cell_y = self.get_cell(px, py)
        # r = max(np.sqrt((px-self.x_centre)**2+(py-self.y_center)**2)*self.alpha,0.8)
        r = self.r
        neighbors = self.get_neighbor_cell(p_cell_x, p_cell_y)
        for pidx in neighbors:
            n_x, n_y = self.sample[pidx]
            distance2 = (n_x - px) ** 2 + (n_y - py) ** 2
            if distance2 < r ** 2:
                # The points are too close, so pt is not a candidate.
                return False
        return True

    def get_point(self, k, px, py):
        # r = max(np.sqrt((px-self.x_centre)**2+(py-self.y_center)**2)*self.alpha,0.8)
        r = self.r
        for i in range(k):
            # rho, theta = np.random.uniform(r, 2 * r), np.random.uniform(0, 2 * np.pi)
            rho, theta = np.random.uniform(r, 1.3 * r), np.random.uniform(0, 2 * np.pi)
            n_px, n_py = np.round(px + rho * np.cos(theta)), np.round(py + rho * np.sin(theta))
            if not (r ** 2 < (n_px - px) ** 2 + (
                    # n_py - py) ** 2 < 4 * r ** 2 and 0 <= n_px < self.y and 0 <= n_py < self.x):
                    n_py - py) ** 2 < (1.3 * r) ** 2 and 0 <= n_px < self.y and 0 <= n_py < self.x):
                continue
            if self.point_valid(n_px, n_py):
                return n_px, n_py
        return False

    def run(self):
        init_px, init_py = self.y // 2, self.x // 2
        self.sample = [(init_px, init_py), ]
        self.activelist.append(0)
        p_c_x, p_c_y = self.get_cell(init_px, init_py)
        self.cell[(p_c_x, p_c_y)] = [0, ]
        idx = 1
        while self.activelist:
            # plt.xlim(0, self.x)
            # plt.ylim(0, self.y)
            samples = self.sample
            # plt.scatter(*zip(*samples), s=6, color='g', alpha=0.1, lw=0)
            pidx = np.random.choice(self.activelist)
            px, py = self.sample[pidx]
            # plt.scatter(px, py, s=6, color='y', alpha=1, lw=0)
            p = self.get_point(self.k, px, py)
            # plt.pause(1)
            if p:
                self.sample.append((p[0], p[1]))
                self.activelist.append(idx)
                p_c_x, p_c_y = self.get_cell(p[0], p[1])
                self.cell[(p_c_x, p_c_y)].append(idx)
                # plt.scatter(p[0], p[1], s=6, color='r', alpha=1, lw=0)
                idx = idx + 1
                if idx % 100 == 0:
                    print('{} get one point, total {}, expectation {:.6f}'.format(self.alpha, idx,
                                                                                  idx / self.x / self.y))
            else:
                self.activelist.remove(pidx)
            # plt.pause(0.1)
        print('expectation:{}'.format(len(self.sample) / (self.x * self.y)))
        # plt.cla()
        return self.sample

    def get_neighbor_cell(self, icx, icy):
        dxdy = [(-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
                (-2, 0), (-1, 0), (1, 0), (2, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
                (-1, 2), (0, 2), (1, 2), (0, 0)]
        neighbours = []
        for cx, cy in dxdy:
            nx, ny = icx + cx, icy + cy
            if not (0 <= nx < self.cell_x_num and 0 <= ny < self.cell_y_num):
                continue
            else:
                neighbours.extend(self.cell[(nx, ny)])
        return neighbours

    # plt.figure()
    # axes = plt.subplot(1, 1, 1)
    # x = np.linspace(-np.pi,np.pi,100)
    # plt.axis('equal')
    # plt.plot(2**0.5*np.sin(x),2**0.5*np.cos(x))
    # plt.plot(2*2**0.5*np.sin(x),2*2**0.5*np.cos(x))
    # plt.plot(4*np.sin(x),4*np.cos(x))
    # plt.xlim(-10,10)
    # plt.ylim(-10,10)
    # plt.xticks(np.arange(-10,10))
    # plt.yticks(np.arange(-10,10))
    # axes.grid(color='gray',linestyle='--',linewidth=1)
    # plt.show()
    # exit(0)
