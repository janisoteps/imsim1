from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import numpy as np

model = VGG19(weights='imagenet', include_top=True)

img_path = 'ling1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

print('Predicted:', decode_predictions(features))
