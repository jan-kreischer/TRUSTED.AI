def save_solution(solution_name, model, training_data, test_data, factsheet, to_webapp=False):
    directory = os.path.join(os.getcwd(), "../solutions", solution_name)
    print(directory)
    if not os.path.exists(directory):
       os.makedirs(directory)
    
    # Persist model
    with open(os.path.join(directory, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    
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
    expcept Exception as e:
        print(e)