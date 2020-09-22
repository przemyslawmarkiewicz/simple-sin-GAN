import numpy as np

from activation_functions import weight_initializer, LeakyReLU, \
    Sigmoid, dSigmoid, dLeakyReLU

from constants import X_DIM, D_HIDDEN, GRADIENT_CLIP, \
    STEP_SIZE_D, WEIGHT_CLIP


class Discriminator(object):
    def __init__(self):
        self.x = None
        self.w1 = weight_initializer(X_DIM, D_HIDDEN)
        self.b1 = weight_initializer(1, D_HIDDEN)
        self.y1 = None
        self.w2 = weight_initializer(D_HIDDEN, D_HIDDEN)
        self.b2 = weight_initializer(1, D_HIDDEN)
        self.y2 = None
        self.w3 = weight_initializer(D_HIDDEN, 1)
        self.b3 = weight_initializer(1, 1)
        self.y3 = None
        self.y = None

    def forward(self, inputs):
        self.x = inputs.reshape(1, X_DIM)
        self.y1 = np.matmul(self.x, self.w1) + self.b1
        self.y1 = LeakyReLU(self.y1)
        self.y2 = np.matmul(self.y1, self.w2) + self.b2
        self.y2 = LeakyReLU(self.y2)
        self.y3 = np.matmul(self.y2, self.w3) + self.b3
        self.y = Sigmoid(self.y3)
        return self.y

    def backward(self, outputs, apply_grads=True):
        # Derivative w.r.t. output
        delta = outputs
        delta *= dSigmoid(self.y)
        # Derivative w.r.t. w3
        d_w3 = np.matmul(np.transpose(self.y2), delta)
        # Derivative w.r.t. b3
        d_b3 = delta.copy()
        # Derivative w.r.t. y2
        delta = np.matmul(delta, np.transpose(self.w3))
        if apply_grads:
            # Update w3
            if np.linalg.norm(d_w3) > GRADIENT_CLIP:
                d_w3 = GRADIENT_CLIP / np.linalg.norm(d_w3) * d_w3
            self.w3 += STEP_SIZE_D * d_w3
            self.w3 = np.maximum(-WEIGHT_CLIP,
                                 np.minimum(WEIGHT_CLIP, self.w3))
            # Update b3
            self.b3 += STEP_SIZE_D * d_b3
            self.b3 = np.maximum(-WEIGHT_CLIP,
                                 np.minimum(WEIGHT_CLIP, self.b3))
        delta *= dLeakyReLU(self.y2)
        # Derivative w.r.t. w2
        d_w2 = np.matmul(np.transpose(self.y1), delta)
        # Derivative w.r.t. b2
        d_b2 = delta.copy()
        # Derivative w.r.t. y1
        delta = np.matmul(delta, np.transpose(self.w2))
        if apply_grads:
            # Update w2
            if np.linalg.norm(d_w2) > GRADIENT_CLIP:
                d_w2 = GRADIENT_CLIP / np.linalg.norm(d_w2) * d_w2
            self.w2 += STEP_SIZE_D * d_w2
            self.w2 = np.maximum(-WEIGHT_CLIP,
                                 np.minimum(WEIGHT_CLIP, self.w2))
            # Update b2
            self.b2 += STEP_SIZE_D * d_b2
            self.b2 = np.maximum(-WEIGHT_CLIP,
                                 np.minimum(WEIGHT_CLIP, self.b2))
        delta *= dLeakyReLU(self.y1)
        # Derivative w.r.t. w1
        d_w1 = np.matmul(np.transpose(self.x), delta)
        # Derivative w.r.t. b1
        d_b1 = delta.copy()
        # Derivative w.r.t. x
        delta = np.matmul(delta, np.transpose(self.w1))
        # Update w1
        if apply_grads:
            if np.linalg.norm(d_w1) > GRADIENT_CLIP:
                d_w1 = GRADIENT_CLIP / np.linalg.norm(d_w1) * d_w1
            self.w1 += STEP_SIZE_D * d_w1
            self.w1 = np.maximum(-WEIGHT_CLIP,
                                 np.minimum(WEIGHT_CLIP, self.w1))
            # Update b1
            self.b1 += STEP_SIZE_D * d_b1
            self.b1 = np.maximum(-WEIGHT_CLIP,
                                 np.minimum(WEIGHT_CLIP, self.b1))
        return delta
