from ClipTokenizer import *
from metrics import loss_function_mse
from stable_diffusion import *
import tensorflow as tf
import numpy as np
import cv2

INITIAL_SENTENCES = [
   #"landscape photo of an unknown new magical breathtaking alien world. dof. bokeh. by artgerm and greg rutkowski. ultra reallistic. extremely detailed. Nikon D850",
   "an unknown new magical breathtaking alien world. dof. bokeh. by artgerm and greg rutkowski. ultra reallistic. extremely detailed. Nikon D850",
   #"landscape photo of an unknown new magical breathtaking alien world.", 
]

def variable_optimization(variable):
    image = np.zeros((1,1,3))
    sd = StableDiffusion(precision_high=True)
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss_fn = lambda: loss_function_mse(variable, image)
    var_list = INITIAL_SENTENCES
    # Optimize for a fixed number of steps
    for _ in range(1000):
        images_result = sd.generate_image(var_list, num_inference_steps=5)
        #opt.minimize(loss_fn, var_list)
        #opt.minimize(loss_fn, var_list)
        print("Len is:",len(images_result))
        print(type(images_result[0]))
        print(loss_function_mse(variable,np.array(images_result[0])))
        

image = cv2.imread("test.jpg")
print(image.shape)
images = variable_optimization(image)

