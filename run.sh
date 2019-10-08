unzip Data/prod_data.zip -d Data/Kiev
mkdir results
python ANN_classification.py Data/Kiev/2017 Data/Kiev/tr_cl.shp Data/Kiev/classification_map.tif
python ind_2_4_1.py 2013 2017 Kiev
