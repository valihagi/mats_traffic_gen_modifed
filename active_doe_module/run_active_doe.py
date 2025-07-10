import json, time, os,sys
from requests.sessions import session
from webapi_client import active_doe_client
from simulation import run_simulation

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--host", metavar="HOSTNAME", default="localhost", help="hostname or ip address of the server")
parser.add_argument("-p", "--port", type=int, default="8011",  help="server port")
parser.add_argument("-q", "--quiet", default=False,  action='store_true', help="do not print infos to the console")
#parser.add_argument("--sg", default=True, action=argparse.BooleanOptionalAction,  help="use service gateway for encryption")
args = parser.parse_args()

with open("setup.json", "r") as fp:
    setup = json.load(fp)

with open("start_design.json", "r") as fp:
    start_design = json.load(fp)

start_data_file="start_data.csv"

n_parallel=1


with active_doe_client(hostname=args.host, port=args.port, use_sg=False) as client:

    # initialize active doe session
    if args.quiet:
        client.silent=True
    session=client.initialize(setup=setup)
    if session is None:
        raise Exception("could not initialize session")

    # insert start data
    if start_data_file is not None:
        client.load_start_data(file_path=start_data_file)

    # run start design
    if len(start_design)>0:
        measurements = [
            dict(
                Variations=variations,
                Responses=run_simulation(variations=variations, setup=setup)
            ) for variations in start_design
        ]
        client.insert_measurements(measurements=measurements)

    # run active doe
    while True:
        # get new candidate recommended by active doe
        candidates=client.get_candidates(size=n_parallel, latest_models_required=True)
        # run simulation responses & insert measurements data to the service
        measurements=[
            dict(
                Index=candidate['Index'],
                Variations=candidate['Variations'],
                Responses=run_simulation(variations=candidate['Variations'], setup=setup)
            ) for candidate in candidates
        ]
        time.sleep(n_parallel) #pause the workflow for readability, remove if not required!
        client.insert_measurements(measurements=measurements)
        # check if stopping is recommended by the service
        if candidates is not None and any([c['Panel']['Algorithm']['StopRecommended'] for c in candidates]):
            break
        
    # finish the test run
    dir_name=os.path.dirname(os.path.abspath(__file__))
    timestamp=time.strftime("%Y_%m_%d-%H%M%S")
    
    #create samples
    samples=client.get_samples(size=500)
    
    samples_file=os.path.join(dir_name, f'samples_{timestamp}.csv')
    client.write_samples(file_path=samples_file, size=500)

    # write result file
    results_file=os.path.join(dir_name, f'test_result_{timestamp}.csv')
    client.write_result(file_path=results_file)
    client.write_result(file_path=start_data_file)
