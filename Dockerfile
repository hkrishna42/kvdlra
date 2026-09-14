FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
ENV PIP_BREAK_SYSTEM_PACKAGES=1 CUDA_HOME=/usr/local/cuda
RUN pip install -q hf_transfer hf_xet ninja numpy scipy matplotlib "kvpress==0.5.1" \
    "transformers==5.8.0" "datasets==2.21.0" "optimum-quanto>=0.2.7" "hqq==0.2.8.post1" "omegaconf>=2.3"
WORKDIR /root/kvdlra
