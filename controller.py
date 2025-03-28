import time
import docker

client = docker.from_env()
adex_container = "adex/cat:latest"
carla_container = "carlasim/carla:0.9.15"

def get_progress():
    try:
        with open("shared/progress.txt", "r") as f:
            return int(f.read().strip())
    except:
        return 0

while True:
    progress = get_progress()
    if progress == 1:
        print("resetting")
        running_containers = client.containers.list()

        # Filter containers by image name
        for container in running_containers:
            image = container.image.tags[0] if container.image.tags else "untagged"
            if image == carla_container:
                print(f"[Controller] Restarting container: {container.name} (Image: {image})")
                container.restart()
                break
        time.sleep(5)
        # To make sure we first start Carla!!
        for container in running_containers:
            image = container.image.tags[0] if container.image.tags else "untagged"
            if image == adex_container:
                print(f"[Controller] Restarting container: {container.name} (Image: {image})")
                container.restart()
                break
    print("going to sleep")
    time.sleep(20)