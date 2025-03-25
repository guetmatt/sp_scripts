# What is this about?

This is the GitHub repository for our data analysis for the Study Project "Affectivity in Political Language", Osnabrück University, Summersemester 2024 - Wintersemeser 24/25. We analyzed narratives in newspaper articles from six of the biggest German national newspapers.

# Files
- ```data``` - Folder with analyzed newspaper articles as textfiles. Filenames in accordance to names in paper.
- ```results``` - Folder with results of analysis as csv-files. Per-article and per-newspaper frequencies (top 50), term frequency-inverse document frequency (tf-idf, top 50), and metadata (token count, type count, etc.)
- ```stats.py``` - Python script with basic processing functions.
- ```analysis.py``` - Python script with user-functions. Runs an analysis and creates files in results-folder on execution. Imports functions from ```stats.py```.

# How to use
- Download or pull the repository and install the ```requirements```.
- To run an analysis and create csv-files in the results folder, execute the following command in your environment: ```python analysis.py```
- Results or data can be viewed more easily when pulled or downloaded to your local machine. 
