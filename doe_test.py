

import json

from active_doe_module.webapi_client import active_doe_client


with open("active_doe_module/setup_xosc.json", "r") as fp:
    setup = json.load(fp)
    
    with active_doe_client(hostname="localhost", port=8011, use_sg=False) as doe_client:
        session=doe_client.initialize(setup=setup)
        if session is None:
            raise Exception("could not initialize session")
        counter = 0
        #client.insert_measurements(measurements=measurements) if we want a kickstarting measurement
        while True:
            counter += .1
            candidates=doe_client.get_candidates(size=1, latest_models_required=True)
            print(candidates)
            measurements = []

            for candidate in candidates:
                traj = None
                ##unpack candiates and insert them into env.scenario
                parameters = candidate["Variations"]
                
                #get KPIS
                kpis = {"min_ttc": counter}
                measurements.append(dict(
                    Index=candidate['Index'],
                    Variations=candidate['Variations'],
                    Responses=kpis)
                )
            doe_client.insert_measurements(measurements=measurements)