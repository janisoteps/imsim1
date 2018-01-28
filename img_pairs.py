from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
import numpy as np
# import pandas as pd
import jsonlines
from keras.models import Model

# Define new model using VGG19 and extracting vector that would be fed into first fully connected layer
# model = VGG16(weights='imagenet', include_top=True)
vgg19 = VGG19(weights='imagenet', include_top=True)
x = vgg19.layers[-1].output
jd_model = Model(input=vgg19.input, output=x)


# Fill in the image hash list
def image_hashes(item_file_path):
    nr_hashes = 0
    img_hashes = []
    with jsonlines.open(item_file_path, 'r') as infile:
        for item in infile:
            pics = item['image_hash']
            for pic in pics:
                img_hashes.append(pic)
                nr_hashes += 1
    return nr_hashes, img_hashes


# Function to encode images to feature vectors and then save to jsonl so that it can be used for similarity training
def image_pairs(item_file_path, img_folder_path):
    # Open file that contains scraped data in jsonline format
    with jsonlines.open(item_file_path, 'r') as infile, jsonlines.open('data/img_pairs.jsonl', mode='w') as outfile:

        # set up counter to limit iterations during development
        count = 1

        while True:
            # Iterate over all objects in jsonline file
            for item in infile:
                pics = item['image_hash']
                # Iterate over image hashes in array
                for pic in pics:
                    # construct image path from hash string and image folder path
                    img_path1 = img_folder_path + '/' + pic + '.jpg'
                    # use Keras preprocessing to load image resized to 224x224px
                    img1 = image.load_img(img_path1, target_size=(224, 224))

                    pic_index = pics.index(pic)
                    print('pic index: ' + str(pic_index))

                    # skip if image is not found
                    if img1 is not None:
                        k = np.random.random()*2
                        if k < 1:
                            label = 1
                            try:
                                pic2 = pics[pic_index + 1]
                                img_path2 = img_folder_path + '/' + pic2 + '.jpg'
                            except:
                                pic2 = pics[pic_index - 1]
                                img_path2 = img_folder_path + '/' + pic2 + '.jpg'
                        else:
                            pic2 = hash_list[int(np.random.random()*hash_nr)]
                            img_path2 = img_folder_path + '/' + pic2 + '.jpg'
                            print('different product image hash: ' + str(pic2))
                            label = 0

                        # use Keras preprocessing to load image resized to 224x224px
                        img2 = image.load_img(img_path2, target_size=(224, 224))

                        if img2 is not None:

                            feature_list1 = feature_list(img1)
                            feature_list2 = feature_list(img2)

                            write_dict = {
                                'label': label,
                                'img_hash1': pic,
                                'img_hash2': pic2,
                                'img_features1': feature_list1,
                                'img_features2': feature_list2
                            }
                            outfile.write(write_dict)

                            print('iteration: ' + str(count))
                            print('label: ' + str(label))
                            print('-----------------------------------')
                            count += 1
                            if count > hash_nr:
                                return


# Obtain path of a random image in dataset
# def image2_path(items, img_folder_path):
#     iterations = int(np.random.random() * 5000)
#     print('different image from line: ' + str(iterations))
#     iterator = 1
#     while True:
#         for item in items:
#             iterator += 1
#             if iterator > iterations:
#                 pics2 = item['image_hash']
#                 try:
#                     pic2 = pics2[int(np.random.random()) * 3]
#                 except:
#                     pic2 = pics2[0]
#                 img_path2 = img_folder_path + '/' + pic2 + '.jpg'
#                 print('second image hash: ' + str(pic2))
#                 return img_path2, pic2


# Calculate feature list
def feature_list(img):
    # Keras preprocess image to an array
    x2 = image.img_to_array(img)
    x2 = np.expand_dims(x2, axis=0)
    x2 = preprocess_input(x2)

    # use VGG19 to generate a feature vector of shape (1, 1000)
    # that's the one before last fully connected layer before softmax layer
    features = jd_model.predict(x2)

    print('image shape: ')
    print(np.shape(features))

    # Transform the feature vector to python list so that it can be written as jsonl
    feat_list = features.tolist()

    return feat_list


# Create image hashes variable
hash_nr, hash_list = image_hashes('/Users/janisdzikevics/dev/scrapers/scraper6/scraper6/spiders/items.jsonl')

# Kick everything off with data file and image folder paths
image_pairs('/Users/janisdzikevics/dev/scrapers/scraper6/scraper6/spiders/items.jsonl', '/Users/janisdzikevics/dev/scrapers/scraper6/scraper6/spiders/images/full')