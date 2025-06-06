# rwse-experiments

Here we present the code used for our experiments with a BERT model for real-world spelling error (RWSE) detection in learner texts.

The prerequisite data cleaning and modifications for our analyses are compiled in the notebook ```data_preparation.ipynb```. 

The notebooks ```analyze_*.ipynb``` contain the analysis pipelines used for each of the five data sets mentioned in our paper.

The code for the magnitude (µ) search is included in the eponymous notebook.

A demo displaying a use case for visualizing RWSEs can be run directly as streamlit app (```demo.py```) or from a ```Dockerfile```.

### External Resources

English news sentences were provided by © 2025 [Universität Leipzig](https://wortschatz.uni-leipzig.de/en/download/English) / Sächsische Akademie der Wissenschaften / InfAI. 

Please visit the GitHub repository of [rwse-checker](https://github.com/zesch/rwse-checker) to view the code for the RWSE detection. 

### Docker
Create an image with name ```rwse_demo``` from within the directory containing ```Dockerfile```.

e.g. ```docker image build --tag rwse_demo .```

Run container from the image ```rwse_demo``` by assigning ```HOST_PORT:CONTAINER_PORT```.

e.g. ```docker run -p 8501:8501 rwse_demo ```