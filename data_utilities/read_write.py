import sys
work_dir = '/content/drive/MyDrive/UCF/CAP7919/Object-Detection/'

def save_object(pickle_name, obj):
    
    if pickle_name is None:
        print("Provide pickle name.")
    
    opened_file_obj = open(work_dir+pickle_name, 'wb')
    pickle.dump(obj, opened_file_ojb)
    opened_file_obj.close()

def load_object(pickle_name):
    
    opened_file_obj = open(work_dir+pickle_name, 'rb')
    loaded = pickle.load(opened_file_obj)
    opened_file_obj.close()
        
    return loaded