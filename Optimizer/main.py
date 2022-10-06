from stable_diffusion import StableDiffusion
import matplotlib.pyplot as plt

stable = StableDiffusion(precision_high=True)
image = stable.generate_image("landscape photo of an unknown new magical breathtaking alien world. dof. bokeh. by artgerm and greg rutkowski. ultra reallistic. extremely detailed. Nikon D850", num_inference_steps=50)
image[0].save("test.jpg")

