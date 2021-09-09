import json

class Factsheet:
    properties = {
        "general": {},
        "fairness": {},
        "explainability": {},
        "robustness": {},
        "methodology": {}  
    }
    
    def __init__(self):
        print("init")
    
    def protected_feature(protected_feature):
        properties["fairness"]["protected_feature"] = protected_feature
    
    def protected_group(protected_group):
        properties["fairness"]["protected_group"] = protected_group
        
    def target_column(target_column):
        properties["general"]["target_column"] = protected_group
    
    def favorable_outcome(favorable_outcome):
        properties["fairness"]["favorable_outcome"] = favorable_outcome
    
    def save(directory):
        with open(os.path.join(directory, "factsheet.json"), 'w') as f:
            json.dump(properties, f)
        
    
        
    
        
        
