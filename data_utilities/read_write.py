import sys
import pickle
work_dir = '/home/abhijit/nat/Object-Detection/'
data_dir = work_dir+'data/'
def save(filename, obj):
    
    if filename is None:
        print("Please provide filename.")
    
    f = open(data_dir+filename, 'wb')
    pickle.dump(obj, f)
    f.close()

def load(filename):
    
    f = open(data_dir+filename, 'rb')
    loaded = pickle.load(f)
    f.close()
        
    return loaded
