import json
import pandas as pd

class DataFrameEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        return json.JSONEncoder.default(self, obj)