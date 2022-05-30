from curses import napms
from re import U
import redis
import pickle as pkl   

class UploadModel:
    def __init__(self, host, port) -> None:       
        self.r = redis.Redis(host=host, port=port, decode_responses=False)
    
    def upload_model(self, model_name, model_path):
        model = pkl.load(open(model_path, 'rb'))
        model_str = pkl.dumps(model)
        self.r.set(model_name, model_str)
        
    def download_model(self, model_name):
        model = self.r.get(model_name)
        model = pkl.loads(model)
        return model