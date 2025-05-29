# rwse-experiments

### Docker
Create an image with name ```rwse_demo``` from within directory containing ```Dockerfile```.

e.g. ```docker image build --tag rwse_demo .```

Run container from the image ```rwse_demo``` by assigning ```HOST_PORT:CONTAINER_PORT```.

e.g. ```docker run -p 8501:8501 rwse_demo ```