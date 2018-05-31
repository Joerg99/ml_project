import pandas as pd
import numpy as np


def generate_dense_vectors():
    data = pd.read_csv('/Users/Jorg/Downloads/ml-latest-small/movies.csv', encoding='utf-8')
    data['index'] = data.index
    lookup = dict(zip(data.movieId, data.index)) # Dictionary Key = movieID, Value = neuer index
    #print(lookup[1])
    #data = data[['index', 'movieId', 'title']]
    #data.to_csv('new_index.csv')
    
    data = np.genfromtxt('/Users/Jorg/Downloads/ml-latest-small/ratings.csv', delimiter=',', usecols=(0,1,2), skip_header=1)
    
    user_ratings = []
    for i in range(1,672):
        user = [0] * 9124
        for entry in data:
            if entry[0] == i:
                user[int(lookup.get(entry[1]))] = entry[2]
        user_ratings.append(user)
    return user_ratings



def get_batch(data, batch_size):
    indices = np.random.randint(0,len(data), batch_size)
    batch = []
    for index in indices:
        batch.append(data[index])
    return batch


if __name__ == '__main__':
    sdasd = generate_dense_vectors()
    print(get_batch(sdasd, 3))
    