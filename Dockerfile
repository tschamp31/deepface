# base image
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

RUN apt-get update && apt-get install -y openssh-server libgl1 protobuf-compiler python3-protobuf python3-filetype ffmpeg libavcodec-dev unixodbc
RUN mkdir /var/run/sshd

# Set root password for SSH access (change 'your_password' to your desired password)
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN useradd -m -s /bin/bash dope
RUN echo "dope:Payson2012!!" | chpasswd

COPY ./Tensorflow-Docker/ffmpeg-deps /home/dope
RUN cd /home/dope/ && apt --fix-broken install

COPY ./opencv_contrib_python_rolling-4.12.0.20250210-cp37-abi3-linux_x86_64.whl /home/dope
RUN pip install /home/dope/opencv_contrib_python_rolling-4.12.0.20250210-cp37-abi3-linux_x86_64.whl
RUN rm /home/dope/opencv_contrib_python_rolling-4.12.0.20250210-cp37-abi3-linux_x86_64.whl

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
