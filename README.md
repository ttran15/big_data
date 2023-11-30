# big_data

### Environment:

```
Ubuntu 22.04
Java JDK 11
Hadoop: 3.3.6
Spark: 3.5.0
```

### Requirement libs:
```
pip install matplotlib
pip install dumpy
pip install pandas
pip install seaborn
pip install geopy.geocoders
pip install folium
pip install findspark
pip install pyspark
pip install statsmodels
pip install numpy
pip install covid-data-api
```

### Files structure:

1. pre_processing_multi.ipynb: using pyspark to pre-process data in multi workers and save to HDFS
2. Data_visualizations: visualize data by all countries, by time, case fatality rate, epidemiological curve, world map
3. model_multi_visualize.ipynb: build ARIMA time series model to predict cases by country
4. Presentation.pptx: slides summarizing what has been done

### How to run:

Step 1: Run pre_processing_multi.ipynb to download raw data from [COVID-19 DATA API](https://pypi.org/project/covid-data-api/) and save preprocessed data to HDFS

Step 2: Run notebooks in Data_visualizations folder to visualize data

Step 3: Run model_multi_visualize.ipynb to build time series model and predict the number of cases by country
