import pandas as pd
import numpy as np
'''
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
data = data[data['counts'] >= 20]
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
data.to_csv('working_data.csv', index=False)
'''

def generate_sparse_vectors():
    data = np.genfromtxt('working_data.csv', dtype='int', delimiter=',', usecols=(0,1,2), skip_header=1, encoding='utf-8')
    user_ratings = []
    
    i = 1
    user = [0] * 1303
    for element in data:
        if element[0] == i:
            user[element[1]] = element[2]
        else:
            i += 1
            user_ratings.append(user)
            user = [0] * 1303 
    return user_ratings
a = generate_sparse_vectors()
def get_batch(data, batch_size):
    indices = np.random.randint(0,len(data), batch_size)
    batch = []
    for index in indices:
        batch.append(data[index])
    return batch

print(get_batch(a, 3))

'''
def generate_dense_vectors():
    data = pd.read_csv('working_data.csv', encoding='utf-8')
    data['index'] = data.index
    #lookup = dict(zip(data.movieId, data.index)) # Dictionary Key = movieID, Value = neuer index
    #print(lookup[1])
    #data = data[['index', 'movieId', 'title']]
    #data.to_csv('new_index.csv')
    
    #data = np.genfromtxt('/Users/Jorg/Downloads/ml-latest-small/ratings.csv', delimiter=',', usecols=(0,1,2), skip_header=1) # user x movies 672x9124
    
    user_ratings = []
    for i in range(1,672):
        user = [0] * 9124
        for entry in data:
            if entry[0] == i:
                user[int(lookup.get(entry[1]))] = entry[2]
        user_ratings.append(user)
    return user_ratings




if __name__ == '__main__':
    sdasd = generate_dense_vectors() # 672x9124
    sdasd = list(map(list, zip(*sdasd))) # 9214 x 672
    
    lengths = []
    for movie in sdasd:
        laenge = 0
        for r in movie:
            if r != 0:
                laenge += 1
        lengths.append(laenge)
    print(lengths)
    
    bad_indices = [idx for idx in range(len(lengths)) if lengths[idx] < 20]
    print(len(bad_indices))
    print(bad_indices)
    for idx in reversed(bad_indices):
        del sdasd[idx]
    print(len(sdasd))
    sdasd = list(map(list, zip(*sdasd))) # 671 x 1303
    sdasd = [[int(j) for j in i]for i in sdasd]
    
    df = pd.DataFrame(sdasd)
    df.to_csv('new_ratings.csv')
'''