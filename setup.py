from setuptools import setup

if __name__ == '__main__':
    setup()
    print('topicnaming installation nearly complete:\n')
    print('If running on a GPU please install llama-cpp-python via:\nCMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python')
    print('If running on a CPU please install llama-cpp-python via:\nCMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python')