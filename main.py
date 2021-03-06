import matplotlib.pyplot as plt
import numpy as np

from generator import Generator
from discriminator import Discriminator
from loss_function import LossFunc

from constants import X_DIM, Z_DIM, ITER_NUM


# get_samples - generates sin based on 10 points (0 to 9) - returns array of sin values
# np.random.uniform(a, b) returns random number between [a, b) in uniform distribution
def get_samples(random=True):
    if random:
        x0 = np.random.uniform(0, 1)
        freq = np.random.uniform(1.2, 1.5)
        mult = np.random.uniform(0.5, 0.8)
    else:
        x0 = 0
        freq = 0.2
        mult = 1
    signal = [mult * np.sin(x0 + freq * i) for i in range(X_DIM)]
    return np.array(signal)


def main():
    G = Generator()
    D = Discriminator()
    criterion = LossFunc()

    real_label = 1
    fake_label = 0

    for itr in range(ITER_NUM):
        # Update D with real data
        x_real = get_samples(True)
        y_real = D.forward(x_real)
        loss_D_r = criterion.forward(y_real, real_label)
        d_loss_D = criterion.backward()
        D.backward(d_loss_D)

        # Update D with fake data
        z_noise = np.random.randn(Z_DIM)
        x_fake = G.forward(z_noise)
        y_fake = D.forward(x_fake)
        loss_D_f = criterion.forward(y_fake, fake_label)
        d_loss_D = criterion.backward()
        D.backward(d_loss_D)

        # Update G with fake data
        y_fake_r = D.forward(x_fake)
        loss_G = criterion.forward(y_fake_r, real_label)
        d_loss_G = D.backward(loss_G, apply_grads=False)
        G.backward(d_loss_G)
        loss_D = loss_D_r + loss_D_f
        if itr % 100 == 0:
            print('{} {} {}'.format(loss_D_r.item((0, 0)),
                                    loss_D_f.item((0, 0)), loss_G.item((0, 0))))

    x_axis = np.linspace(0, 10, 10)
    for i in range(50):
        z_noise = np.random.randn(Z_DIM)
        x_fake = G.forward(z_noise)
        plt.plot(x_axis, x_fake.reshape(X_DIM))
    plt.ylim((-1, 1))
    plt.show()


if __name__ == '__main__':
    main()