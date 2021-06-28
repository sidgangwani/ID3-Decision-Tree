# ID3-Decision-Tree

For this project, I created a ID3 decision tree in google collab using python which accepted a dataset of train_data which was used to implement our ID3 decision tree to check the edibility of a mushroom. A test_dataset was then accepted to predict whether the mushroom was edible or poisonous. 

The accuracy that I got was 1.00 and I got the following info gain values for the following root nodes:

5(odor) = 0.9137211754508912
20(spore) = 0.1328617458581757
22(habitat) = 0.2354746344713046
8(gill-size) = 0.7062740891876004
3(cap-color) = 0.7300166301457934

My ID3 decision tree:

{5: {'a': 'e',
     'c': 'p',
     'f': 'p',
     'l': 'e',
     'm': 'p',
     'n': {20: {'b': 'e',
                'h': 'e',
                'k': 'e',
                'n': 'e',
                'o': 'e',
                'r': 'p',
                'w': {22: {'d': {8: {'b': 'e', 'n': 'p'}},
                           'g': 'e',
                           'l': {3: {'c': 'e', 'n': 'e', 'w': 'p', 'y': 'p'}},
                           'p': 'e',
                           'w': 'e'}},
                'y': 'e'}},
     'p': 'p',
     's': 'p',
     'y': 'p'}}

**To run the google colab file. Open google colab and then select file, open notebook, select upload and upload the downloaded google colab file from the zip file.**
