import os
import json

class Factsheet:
    properties = {
        "general": {},
        "fairness": {},
        "explainability": {},
        "robustness": {},
        "methodology": {}  
    };
    
    def __init__(self):
        print("init")
    
    def set_protected_feature(self, protected_feature):
        self.properties["fairness"]["protected_feature"] = protected_feature
    
    def set_protected_group(self, protected_group):
        self.properties["fairness"]["protected_group"] = protected_group
        
    def set_target_column(self, target_column):
        self.properties["general"]["target_column"] = target_column
    
    def set_favorable_outcome(self, favorable_outcome):
        self.properties["fairness"]["favorable_outcome"] = favorable_outcome
    
    def save(self, directory):
        with open(os.path.join(directory, "factsheet.json"), 'w') as f:
            json.dump(self.properties, f)
        
    
        
    
        
        
