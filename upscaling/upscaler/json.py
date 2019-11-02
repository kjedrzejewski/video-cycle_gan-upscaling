import json
import pandas as pd
import numpy as np

class PandasEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        if isinstance(obj, pd.Series):
            return obj.tolist()
        
        if isinstance(obj, np.integer):
            return int(obj)
        
        return json.JSONEncoder.default(self, obj)