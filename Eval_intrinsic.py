import pandas as pd
import numpy as np
import Autoencoder_masked_loss
import heapq
 
predictions_only = np.subtract(Autoencoder_masked_loss.results, Autoencoder_masked_loss.x_test)
x_test = np.array(Autoencoder_masked_loss.x_test)
index_of_top_predictions = []
index_of_top_ratings = []
for l in predictions_only:
    index_of_top_predictions.append(heapq.nlargest(10, range(len(l)), l.take))
for k in x_test:
    index_of_top_ratings.append(heapq.nlargest(10, range(len(k)), k.take))
 
movie_lookup = pd.read_csv('working_data_short.csv', encoding='utf-8')
movie_lookup = movie_lookup.drop(['userId', 'rating'], axis=1)
movie_lookup = movie_lookup.sort_values('movieId')
movie_lookup = movie_lookup.drop_duplicates()
movie_lookup = movie_lookup.set_index('movieId')
lookup_dict = movie_lookup.to_dict('index')

for user_predictions, user_ratings in zip(index_of_top_predictions, index_of_top_ratings):
    for pred, rating in zip(user_predictions, user_ratings):
        print(lookup_dict.get(pred).get('name'),'\t \t' ,lookup_dict.get(rating).get('name'))
    print('_______')
