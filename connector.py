import json
import os
import time

from active_doe_module.webapi_client import active_doe_client

def get_kpi():
    try:
        with open("shared/doe_messages/kpi.json", "r") as f:
            return json.load(f)
    except:
        return None
    
def clear_kpi():
    try:
        with open("shared/doe_messages/kpi.json", "w") as f:
            json.dump([], f)
    except:
        return None

def write_candidates(data):
    with open("shared/doe_messages/candidates.json", "w") as f:
        json.dump(data, f)

    
with open("active_doe_module/setup_xosc.json", "r") as fp:
        setup = json.load(fp)



json_file_number = 7

logs = "doe_logs/"

json_file = f"{logs}meas{json_file_number}.json"
samples_file = f"{logs}samples{json_file_number}.json"




with active_doe_client(hostname="localhost", port=8011, use_sg=False) as doe_client:
    session=doe_client.initialize(setup=setup)
    if session is None:
        raise Exception("could not initialize session")
    counter = 0
    
    counter = 0
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        doe_client.insert_measurements(measurements=data)
    
    while True:
        counter += 1
        while True:
            candidates=doe_client.get_candidates(size=1, latest_models_required=True)
            if candidates is not None:
                break
            time.sleep(10)

        if (candidates is not None and any([c['Panel']['Algorithm']['StopRecommended'] for c in candidates])) or counter > 200:
            print(f"Model building finished, samples can be found in {samples_file}.")
            samples = doe_client.get_samples(size=150)
            with open(samples_file, "w") as f:
                json.dump(samples, f, indent=2)
            results_file=os.path.join(logs, f'test_result_doe.csv')
            doe_client.write_result(file_path=results_file)
            write_candidates([1])
            exit()

        write_candidates(candidates)

        print("going to sleep")
        time.sleep(50)


        while True:
            kpi_data = get_kpi()
            print(kpi_data)
            if kpi_data is None or kpi_data == []:
                time.sleep(10)
                print("continue")
                continue
            break
            
        print(f"inserting meas {kpi_data}")
        doe_client.insert_measurements(measurements=kpi_data)
        clear_kpi()