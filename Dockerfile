# base image
FROM nvcr.io/nvidia/tensorflow:25.04-tf2-py3

RUN apt-get update && apt-get install -y openssh-server libgl1 protobuf-compiler python3-protobuf python3-filetype ffmpeg libavcodec-dev unixodbc python3-venv sudo

COPY ./Tensorflow-Docker/ffmpeg-deps /home/dope
RUN cd /home/dope/ && apt --fix-broken install

# Now replaced with CVCuda
# COPY Tensorflow-Docker/opencv_contrib_python_rolling-4.12.0.86-cp312-cp312-linux_x86_64.whl /home/dope
# RUN pip install /home/dope/opencv_contrib_python_rolling-4.12.0.86-cp312-cp312-linux_x86_64.whl
# RUN rm /home/dope/opencv_contrib_python_rolling-4.12.0.86-cp312-cp312-linux_x86_64.whl

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
