import json
import os

samples = doe_client.get_samples(size=200)

with open(samples_file, "w") as f:
    json.dump(samples, f, indent=2)

results_file = os.path.join(logs, 'test_result_doe.csv')
doe_client.write_result(file_path=results_file)

write_candidates([1])

print("[Injected] Successfully dumped samples, wrote results, and wrote candidates.")
