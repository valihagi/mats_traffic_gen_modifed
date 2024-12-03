# Start from the official ROS 2 Humble base image
FROM ros:humble-ros-base

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    docker.io \
    xauth \
    python3-rocker \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /mats-trafficgen

# Copy your application files
COPY ./cat /mats-trafficgen/cat
COPY ./requirements.txt /mats-trafficgen/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r ./cat/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir Cython
RUN pip3 install --no-cache-dir docker

# Create a symlink for numpy (if needed)
RUN ln -s /usr/local/lib/python3.10/site-packages/numpy/core/include/numpy/ /usr/include/numpy 

# Compile Cython files
RUN cd /mats-trafficgen/cat/advgen && cythonize -i -a utils_cython.pyx

# Copy the rest of your application files
COPY . /mats-trafficgen

# Set the Python path
ENV PYTHONPATH "${PYTHONPATH}:/mats-trafficgen/cat"

# Optionally source the ROS 2 environment
# Note: You can remove this if you don't need ROS 2 in your entrypoint script
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Set the command to run your Python script
CMD ["bash"]

