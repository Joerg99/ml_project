import pandas as pd
import numpy as np
'''
############# code for creating new csv file ###################
data = pd.read_csv('/Users/Jorg/Downloads/ml-latest-small/ratings.csv', encoding='utf-8')
mov_names = pd.read_csv('/Users/Jorg/Downloads/ml-latest-small/movies.csv', encoding='utf-8')

data['index'] = data.index
movie_occ = data.groupby(['movieId'])['movieId'].count()
movie_occ = movie_occ.to_frame()
movie_occ.columns = ['counts']
movie_occ['movIndex'] = movie_occ.index
data['counts'] = data['movieId'].map(movie_occ.set_index('movIndex')['counts'])
data['name'] = data['movieId'].map(mov_names.set_index('movieId')['title'])


#print(data.loc[data['movieId'] == 31])
data = data[data['counts'] >= 80] # Filme mit ueber x Bewertungen bleiben drin
#print(data)
uni = data.movieId.unique()
lookup = dict(zip(sorted(uni), [x for x in range(len(uni))]))
#print(lookup)
lookup = pd.DataFrame.from_dict(lookup, orient='index')
lookup['index'] = lookup.index
lookup.columns = ['new_index','index']
data['movieId'] = data['movieId'].map(lookup.set_index('index')['new_index'])
data = data.drop(['index', 'timestamp', 'counts'], axis=1)
data['rating'] = data.rating.astype(int)
data.to_csv('working_data_short.csv', index=False)
'''

def generate_sparse_vectors():
    data = np.genfromtxt('working_data_short.csv', dtype='int', delimiter=',', usecols=(0,1,2), skip_header=1, encoding='utf-8')
    user_ratings = []
    llist = [l[1] for l in data]
    len_user_vector = (max(llist)) +1
    
    i = 1
    user = [0] * len_user_vector
    for element in data:
        if element[0] == i:
            user[element[1]] = element[2]
        else:
            i += 1
            user_ratings.append(user)
            user = [0] * len_user_vector 
    return user_ratings

a = generate_sparse_vectors()

def get_batch(data, batch_size):
    indices = np.random.randint(0,len(data), batch_size)
    batch = []
    for index in indices:
        batch.append(data[index])
    return batch



if __name__ == '__main__':
    b = get_batch(a, 3)
    print(len(b[1]))