import json
import pandas as pd

class PandasEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        if isinstance(obj, pd.Series):
            return obj.tolist()
        
        return json.JSONEncoder.default(self, obj)