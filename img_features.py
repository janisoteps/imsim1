from vgg16jd import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input
import numpy as np
# import pandas as pd
import jsonlines
from keras.models import Model

# Define new model using VGG16 and extracting vector that would be fed into first fully connected layer
# This gets rid of VGG16 fully connected layers so that I can train my own for triplet loss
# model = VGG16(weights='imagenet', include_top=True)
vgg16 = VGG16(weights='imagenet', include_top=True)
x = vgg16.layers[-2].output
jd_model = Model(input=vgg16.input, output=x)


# Function to encode images to feature vectors and then save to jsonl so that it can be used for similarity training
def image_encode(item_file_path, img_folder_path):
    # Open file that contains scraped data in jsonline format
    with jsonlines.open(item_file_path, 'r') as infile, jsonlines.open('features.jsonl', mode='w') as outfile:

        # set up counter to limit iterations during development
        count = 0

        while True:
            # Iterate over all objects in jsonline file
            for item in infile:
                pics = item['image_hash']
                # Iterate over image hashes in array
                for pic in pics:
                    # construct image path from hash string and image folder path
                    img_path = img_folder_path + '/' + pic + '.jpg'
                    # use Keras preprocessing to load image resized to 224x224px
                    img = image.load_img(img_path, target_size=(224, 224))

                    # skip if image is not found
                    if img is not None:
                        # Keras preprocess image to an array
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)

                        # use VGG16 to generate a feature vector of shape (1, 4096)
                        # that's the one before last fully connected layer before softmax layer
                        # features = model.predict(x)
                        features = jd_model.predict(x)

                        print(np.shape(features))
                        # pandas dataframe can accept only 2d inputs
                        #  we are ok with that for fully connected training later
                        # features = np.reshape(features, (1, 25088))
                        # print(np.shape(features))

                        # features_df = pd.DataFrame(features)
                        # features_df.to_csv(outfile)

                        # Transform the feature vector to python list so that it can be written as jsonl
                        feature_list = features.tolist()
                        write_dict = {'img_hash': pic, 'img_features': feature_list}
                        outfile.write(write_dict)

                        count += 1
                        print(count)
                        if count > 999:
                            return


image_encode('/Users/janisdzikevics/dev/scrapers/scraper6/scraper6/spiders/items.jsonl', '/Users/janisdzikevics/dev/scrapers/scraper6/scraper6/spiders/images/full')