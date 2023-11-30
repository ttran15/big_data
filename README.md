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
matplotlib==3.8.2
dumpy==0.1.2
pandas==2.1.3
seaborn==0.13.0
geopy==2.4.1
folium==0.15.0
findspark==2.0.1
pyspark==3.5.0
statsmodels==0.14.0
numpy==1.26.2
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
