# big_data

Environment:

```
Ubuntu 22.04
Java JDK 11
Hadoop: 3.3.6
Spark: 3.5.0
```

Requirement libs:
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

File structure:

1. pre_processing_multi: using pyspark to pre-process data in multi workers
2. Data_visualizations: visualize data
3. model_multi_visualize: build ARIMA time series model to predict cases by country
4. Presentation: slides summarizaing what we have done
