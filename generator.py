import numpy as np

from activation_functions import weight_initializer, ReLU, \
    Tanh, dTanh, dReLU

from constants import GRADIENT_CLIP, Z_DIM, G_HIDDEN, X_DIM, \
    STEP_SIZE_G, WEIGHT_CLIP


class Generator(object):
    def __init__(self):
        self.z = None
        self.w1 = weight_initializer(Z_DIM, G_HIDDEN)
        self.b1 = weight_initializer(1, G_HIDDEN)
        self.x1 = None
        self.w2 = weight_initializer(G_HIDDEN, G_HIDDEN)
        self.b2 = weight_initializer(1, G_HIDDEN)
        self.x2 = None
        self.w3 = weight_initializer(G_HIDDEN, X_DIM)
        self.b3 = weight_initializer(1, X_DIM)
        self.x3 = None
        self.x = None

    def forward(self, inputs):
        self.z = inputs.reshape(1, Z_DIM)
        self.x1 = np.matmul(self.z, self.w1) + self.b1
        self.x1 = ReLU(self.x1)
        self.x2 = np.matmul(self.x1, self.w2) + self.b2
        self.x2 = ReLU(self.x2)
        self.x3 = np.matmul(self.x2, self.w3) + self.b3
        self.x = Tanh(self.x3)
        return self.x

    def backward(self, outputs):
        # Derivative w.r.t. output
        delta = outputs
        delta *= dTanh(self.x)
        # Derivative w.r.t. w3
        d_w3 = np.matmul(np.transpose(self.x2), delta)
        # Derivative w.r.t. b3
        d_b3 = delta.copy()
        # Derivative w.r.t. x2
        delta = np.matmul(delta, np.transpose(self.w3))
        # Update w3
        if (np.linalg.norm(d_w3) > GRADIENT_CLIP):
            d_w3 = GRADIENT_CLIP / np.linalg.norm(d_w3) * d_w3
        self.w3 -= STEP_SIZE_G * d_w3
        self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w3))
        # Update b3
        self.b3 -= STEP_SIZE_G * d_b3
        self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b3))
        delta *= dReLU(self.x2)
        # Derivative w.r.t. w2
        d_w2 = np.matmul(np.transpose(self.x1), delta)
        # Derivative w.r.t. b2
        d_b2 = delta.copy()

        # Derivative w.r.t. x1
        delta = np.matmul(delta, np.transpose(self.w2))

        # Update w2
        if (np.linalg.norm(d_w2) > GRADIENT_CLIP):
            d_w2 = GRADIENT_CLIP / np.linalg.norm(d_w2) * d_w2
        self.w2 -= STEP_SIZE_G * d_w2
        self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w2))

        # Update b2
        self.b2 -= STEP_SIZE_G * d_b2
        self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b2))
        delta *= dReLU(self.x1)
        # Derivative w.r.t. w1
        d_w1 = np.matmul(np.transpose(self.z), delta)
        # Derivative w.r.t. b1
        d_b1 = delta.copy()

        # No need to calculate derivative w.r.t. z
        # Update w1
        if (np.linalg.norm(d_w1) > GRADIENT_CLIP):
            d_w1 = GRADIENT_CLIP / np.linalg.norm(d_w1) * d_w1
        self.w1 -= STEP_SIZE_G * d_w1
        self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w1))

        # Update b1
        self.b1 -= STEP_SIZE_G * d_b1
        self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b1))
