import unittest

import keras
import numpy as np
from tensorflow.keras.layers import LSTM
import tensorflow.compat.v2 as tf
import BinaryLSTM
import importlib
import larq

class Test_ste_sign(unittest.TestCase):
    def test_binarize(self):
        #does binarization work
        calc = larq.quantizers.ste_sign(np.array([[1.6, -1.6],[0.2, -0.2],[-1.0, 1.0], [0, -0]])).numpy()
        truth = np.array([[ 1., -1.], [ 1., -1.], [-1.,  1.], [ 1.,  1.]])
        equal = np.equal(calc, truth)
        self.assertTrue(np.all(equal))

    def test_gradient_simple(self):
        #does straight through estimator work
        x = tf.Variable([1.5, -1.5, 0.5, -0.5])
        with tf.GradientTape() as tape:
            y = larq.quantizers.ste_sign(x)
        dy_dx = tape.gradient(y, x).numpy()
        #according to the straight through estimator, gradient = x if (|x|<1) else 0
        truth = np.array([0.0, 0.0, 1.0, 1.0])
        equal = np.equal(dy_dx, truth)
        self.assertTrue(np.all(equal))

    def test_gradient_complex(self):
        #does straight through work with other funcs
        x = tf.Variable([1.5, -1.5, 0.5, -0.5])
        with tf.GradientTape() as tape:
            y = larq.quantizers.ste_sign(x)
            z = y**2
        dz_dx = tape.gradient(z, x).numpy()
        truth = np.array([0.0, 0.0, 2.0, -2.0])
        equal = np.equal(dz_dx, truth)
        self.assertTrue(np.all(equal))

def test_clip_optim(optim):
        opt = optim(learning_rate=1, clipvalue=1)
        opt2 = optim(learning_rate=1)
        var = tf.Variable([1.0, -1.0, 2.0, -2.0])
        var2 = tf.Variable([1.0, -1.0, 2.0, -2.0])
        # dy/dx = x for x**2/2, so clipvalue should affect last 2 variables, not first two. 
        # Additionally the changes in last to variables should be same as first two, since clipping makes their gradients equal
        loss = lambda: (var ** 2)/2     # d(loss)/d(var1) = var1
        loss2 = lambda: (var2 **2)/2
        opt.minimize(loss, [var])
        opt2.minimize(loss2, [var2])
        return var, var2

class Test_clipvalue(unittest.TestCase):
    # we might not have to use this? unsure if this is mentioned in the papers
    def test_clipvalue_rmsprop(self):
        var, var2 = test_clip_optim(tf.keras.optimizers.experimental.RMSprop)
        self.assertTrue(np.all(var.numpy()[:2] == var.numpy()[2:] + np.array([-1, 1])))
        self.assertTrue(np.all(var.numpy()[:2] == var2.numpy()[:2]))
        self.assertFalse(np.any(var.numpy()[2:] == var2.numpy()[2:]))

    def test_clipvalue_adam(self):
        var, var2 = test_clip_optim(tf.keras.optimizers.experimental.Adam)
        self.assertTrue(np.all(var.numpy()[:2] == var.numpy()[2:] + np.array([-1, 1])))
        self.assertTrue(np.all(var.numpy()[:2] == var2.numpy()[:2]))
        self.assertFalse(np.any(var.numpy()[2:] == var2.numpy()[2:]))
    
    def test_clipvalue_sgd_precise(self):
        var, var2 = test_clip_optim(tf.keras.optimizers.experimental.SGD)
        true_var = np.array([ 0.,  0.,  1., -1.])
        true_var2 = np.array([0., 0., 0., 0.])
        self.assertTrue(np.all(var.numpy() == true_var))
        self.assertTrue(np.all(var2.numpy() == true_var2))


class Test_weightclip(unittest.TestCase):
    def test_weightclip_simple(self):
        org = np.array([-1.5, -0.5, 0.5, 1.5])
        true_clip = np.array([-1. , -0.5,  0.5,  1. ])
        clip = larq.constraints.weight_clip(clip_value=1)(org).numpy()
        self.assertTrue(np.all(clip==true_clip))
    
    def test_weightclip_complex(self):
        tf.keras.utils.set_random_seed(1)
        seq_length=5
        X =[[i+j for j in range(seq_length)] for i in range(100)]
        X =np.array(X)
        y =[[ i+(i-1)*.5+(i-2)*.2+(i-3)*.1 for i in range(4,104)]]
        y =np.array(y)
        X=X.reshape((100,5,1))
        y=y.reshape((100,1))
        model = keras.models.Sequential()
        model.add(BinaryLSTM.BinaryLSTM(8,input_shape=(5,1), return_sequences=False, 
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2.0, seed=1),
                                        recurrent_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2.0, seed=1),
                                        bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2.0, seed=1),
                                        gamma_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2.0, seed=1),
                                        recurrent_constraint=larq.constraints.weight_clip(clip_value=1), 
                                        kernel_constraint=larq.constraints.weight_clip(clip_value=1)))
        model.add(keras.layers.Dense(2,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1), activation='linear'))
        model.add(keras.layers.Dense(1,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1), activation='linear'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        model.compile(loss='mse',optimizer = opt, metrics=['MSE'])
        model.fit(X, y, epochs=1, batch_size=5, verbose=0)
        self.assertTrue(np.all((model.get_weights()[0] >= -1)*(model.get_weights()[0] <= 1)))
        self.assertTrue(np.all((model.get_weights()[1] >= -1)*(model.get_weights()[1] <= 1)))

if __name__ == '__main__':
    unittest.main()