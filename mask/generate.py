from __future__ import print_function
import numpy as np
import os, cv2, threading
from poisson_disc import VD_Poisson_disc
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


def binomial(prob):
    prob_vec = np.reshape(prob, [-1, ])
    mask = np.random.binomial(size=prob_vec.size, n=1, p=prob_vec)
    mask = np.reshape(mask, prob.shape)
    return mask


def gaussian_prob(r, sigma, dist2):
    x2 = dist2
    z = 2 * r / (np.sqrt(2 * np.pi) * sigma) * np.exp(- x2 / (2 * sigma ** 2))
    z = max(z, 0)
    return z


def square_prob(r, sigma, dist2):
    x2 = dist2
    z = - x2 * r / (3 * np.sqrt(2 * np.pi) * sigma ** 4) + 2 * r / (np.sqrt(2 * np.pi) * sigma)
    z = max(z, 0)
    return z


def threadfun(hei, wid, alpha, radius, k):
    print('start alpha={}, radius={}, k={}'.format(alpha, radius, k))
    # generate poisson disc point collection
    poi_disc = VD_Poisson_disc(hei, wid, alpha, radius, k)
    #  1 0.5
    poi_disc.run()

    # points to image
    img = np.zeros((hei, wid), dtype=np.uint8)
    for px, py in poi_disc.sample:
        img[int(px), int(py)] = 255

    sample_rate = len(poi_disc.sample) * 1.0 / hei / wid
    save_dir = str(hei) + 'x' + str(wid)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(
        '{}/VD_poisson_disc_alpha{:.5f}_radius{:4f}_rate{:.4f}.png'.format(save_dir, alpha, radius, sample_rate), img)
    print('Save {}/VD_poisson_disc_alpha{:.5f}_radius{:4f}_rate{:.4f}.png'.format(save_dir, alpha, radius, sample_rate))
    print('{}x{}: alpha:{}, radius:{}, sample rate:{}'.format(hei, wid, alpha, radius, sample_rate))
    # cv2.imwrite('{}/VD_poisson_disc_alpha{:.5f}_rate{:.4f}.png'.format(save_dir, alpha, sample_rate), img)
    # print('Save {}/VD_poisson_disc_alpha{:.5f}_rate{:.4f}.png'.format(save_dir, alpha, sample_rate))
    # print('{}x{}: alpha:{}, sample rate:{}'.format(hei, wid, alpha, sample_rate))


if __name__ == '__main__':

    ##  r increase as point (x,y) far away from center, alpha controls the increasing rate

    # 320x320
    # 10% 0.029 0.1053
    # alpha_list = [0.024,0.01,0.015,0.02,0.005,0.008,0.016,0.017,0.018,0.019,0.011,0.012,0.013,0.014]

    # 256x256
    # percent   alpha   rate
    # 5%        0.0573   0.0499
    # 10%       0.0375   0.1000
    # 20%       0.0239   0.2004
    # 30%       0.01813  0.3002
    # 40%       0.01475  0.4000
    # 50%       0.01245  0.5008
    alpha_list = [0.05, ]
    rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    min_prob = rate_list / np.sqrt(2 * np.pi)  # [0.0399, 0.0798, 0.1197, 0.1596, 0.1995]
    sigma_list = [0.2660, ]
    # min_prob = 0.5151 * radius ^ (-1.76)
    # radius_list = [4.0, 2.90, 2.30, 2.10, 1.60]
    radius_list = [2.3,]

    # rate = 0.504 * np.power(radius, -1.68)

    # hei, wid = 320, 320
    hei, wid = 256, 256

    ######################
    # k = 64
    # thread = []
    # for radius in radius_list:
    #     for alpha in alpha_list:
    #         t = threading.Thread(target=threadfun, args=(hei, wid, alpha, radius, k))
    #         thread.append(t)
    #         t.start()
    #     for t in thread:
    #         t.join()
    # print('Done')
    ######################
    base_mask_list = [
        './256x256/VD_poisson_disc_alpha0.05000_radius4.087600_rate0.0401.png'
        './256x256/VD_poisson_disc_alpha0.05000_radius2.900000_rate0.0783.png'
        './256x256/VD_poisson_disc_alpha0.05000_radius2.300000_rate0.1250.png'
        './256x256/VD_poisson_disc_alpha0.05000_radius2.200000_rate0.1599.png'
        './256x256/VD_poisson_disc_alpha0.05000_radius2.000000_rate0.2027.png'
    ]
    ######################
    # # for bask_mask_name in base_mask_list:
    # idx = 0
    # rate = rate_list[idx]
    # sigma = sigma_list[idx]
    # prob = np.zeros(shape=[hei, wid], dtype='float')
    # c_x, c_y = hei / 2, wid / 2
    # for x in range(hei):
    #     for y in range(wid):
    #         dist2 = (float(x - c_x) / c_x) ** 2 + (float(y - c_y) / c_y) ** 2
    #         prob[x, y] = gaussian_prob(rate, sigma, dist2)
    #         # prob[x, y] = square_prob(rate, sigma, dist)
    #         prob[x, y] = max(prob[x, y], min_prob[idx])
    #
    # mask = binomial(prob)
    #
    # plt.figure(figsize=(8, 8), dpi=100)
    # plt.title('Probability')
    # plt.subplot(2, 2, 1)
    # fig_obj = plt.imshow(prob, cmap=plt.get_cmap('jet'))
    # plt.colorbar(fig_obj)
    # plt.title('Probability (avg=%.4f)' % np.mean(prob))
    # plt.subplot(2, 2, 2)
    # fig_obj = plt.imshow(mask, cmap=plt.get_cmap('gray'))
    # plt.colorbar(fig_obj)
    # plt.title('Mask (%.2f%%)' % (100.0 * np.sum(mask) / mask.size))
    # plt.subplot(2, 2, 3)
    # # fig_obj = plt.plot(prob)
    # # plt.title('PDF')
    # fig_obj = plt.plot(np.mean(prob, axis=0))
    # plt.grid(True)
    # plt.title('PDF (Row)')
    # plt.subplot(2, 2, 4)
    # # fig_obj = plt.plot(prob)
    # fig_obj = plt.plot(np.mean(prob, axis=1))
    # plt.grid(True)
    # plt.title('PDF (Col)')
    # plt.tight_layout()
    # plt.savefig(os.path.join('Parametric_mask.png'))
    # print('Saving figure at ' + os.path.join('Parametric_mask.png'))
    # plt.show(block=False)
    # plt.pause(0.01)
