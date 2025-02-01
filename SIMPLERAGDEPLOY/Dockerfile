#use the AWS base image for python 3.12
FROM public.ecr.aws/lambda/python:3.12

#install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

#COPY the requirements.txt file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

#Install the libs in requirements.txt
RUN pip install -r requirements.txt

#Copy PDF File
COPY Kanastra_Documentation.pdf ${LAMBDA_TASK_ROOT}/Kanastra_Documentation.pdf

#COPY the function code
COPY simplerag.py ${LAMBDA_TASK_ROOT}

#Set permission to make the file executable
RUN chmod +x simplerag.py

#Set CMD to your handler
CMD ["simplerag.lambda_handler"]