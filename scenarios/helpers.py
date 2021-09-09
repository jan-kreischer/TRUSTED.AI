import os
import pickle
import pandas as pd

def save_solution(scenario_id, solution_id, model, training_data, test_data, factsheet, to_webapp=False):
    if to_webapp:
        # save to the webapps scenario folder
        directory = os.path.join(os.getcwd(), "solutions", solution_id)
#"../webapp/scenarios"
    else:
        # save to the main scenarios folder
        directory = os.path.join(os.getcwd(), "solutions", solution_id)
    print("base directory {}".format(directory))

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Persist model
    try:
        with open(os.path.join(directory, "model.pkl"), 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        print(e)
    
    # Persist training_data
    try:
        training_data.to_csv(os.path.join(directory, "train.csv"), index=False)
    except Exception as e:
        print(e)
        
    # Persist test_data
    try:
        test_data.to_csv(os.path.join(directory, "test.csv"), index=False)
    except Exception as e:
        print(e)         
        
    try:
        factsheet.save(directory)
    except Exception as e:
        print(e)