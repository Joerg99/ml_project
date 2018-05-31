import numpy as np
import matplotlib.pyplot as plt


def make_data(seq_length, batch_size):
    result = np.zeros((batch_size, seq_length))
        
    for i in range(len(result)):
        result[i] = np.random.randint(4 , size=seq_length)#list(range(seq_length)) #/2)) + list(reversed(range((seq_length/2) )))

    return result

a = make_data(100, 12)
plt.imshow(a)
plt.show()

