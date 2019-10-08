import gdal
import osr
import numpy
import ogr
import sys
import os
import math
import pickle
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time
import argparse
def read(path,name=''):
    list_tifs=os.listdir(path)
    list_del=[]
    for i in list_tifs:
        if i.rfind(name)==-1 or i.rfind(".tif")==-1 or i.rfind(".tif")-len(i)!=-4:
            list_del.append(i)
    for i in list_del:
        list_tifs.remove(i)
    return list_tifs 

def Joint_Raster(path,name):
    raster_list=read(path,name)
    Rasters=[]
    X0BigRaster=0
    Y0BigRaster=0
    XEBigRaster=0
    YEBigRaster=0
    PixelWidth=0
    PixelHeight=0
    bandsnum=0
    raster_srs = osr.SpatialReference()
    for i in raster_list:
        a=gdal.Open(os.path.join(path,i))
        transform=a.GetGeoTransform()
        xOrigin = transform[0]
        yOrigin = transform[3]
        xsize = a.RasterXSize
        ysize = a.RasterYSize
        if raster_list.index(i)==0:
            PixelWidth=transform[1]
            PixelHeight=transform[5]
        xLast=xOrigin+xsize*PixelWidth + PixelWidth
        yLast=yOrigin+ysize*PixelHeight + PixelHeight
        if raster_list.index(i)==0:
            raster_srs.ImportFromWkt(a.GetProjectionRef())
            X0BigRaster=xOrigin
            Y0BigRaster=yOrigin
            XEBigRaster=xLast
            YEBigRaster=yLast
        if X0BigRaster<xOrigin:
            X0BigRaster=xOrigin
        if Y0BigRaster>yOrigin:
            Y0BigRaster=yOrigin
        if XEBigRaster>xLast:
            XEBigRaster=xLast
        if YEBigRaster<yLast:
            YEBigRaster=yLast
        bandsnum=bandsnum+a.RasterCount
        Rasters.append([a,xOrigin,yOrigin,xsize,ysize,xLast,yLast,a.RasterCount])
    x_size=int(math.fabs(XEBigRaster-X0BigRaster)/PixelWidth)
    y_size=int(math.fabs((Y0BigRaster-YEBigRaster)/(-PixelHeight)))
    Rasters2=[]
    for R in Rasters:
        x0=R[1]
        y0=R[2]
        x0_c=int(math.fabs((x0-X0BigRaster)/PixelWidth))
        y0_c=int(math.fabs((y0-Y0BigRaster)/PixelHeight))
        Rasters2.append([R[0],x0_c,y0_c,R[7]])
    class tif_param:
       Rasters = Rasters2
       X0 = X0BigRaster
       Y0 = Y0BigRaster
       xsize = x_size-1
       ysize = y_size-1
       bands_num = bandsnum
       pixel_width = PixelWidth
       pixel_height = PixelHeight
    return tif_param

def field_paste(ff,num):
    res=numpy.empty((num,1))
    for i in range(res.shape[0]):
        res[i][0]=ff
    return res

def zonal_stats(feat,input_zone_polygon,pixelWidth,pixelHeight,X0_Big,Y0_Big,Rasters,xsize,ysize,bands_num):
    raster = Rasters[0][0]
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    # Get raster georeference info
    xOrigin = X0_Big
    yOrigin = Y0_Big
    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    #feat = lyr.GetNextFeature()
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)
    # Get extent of feat
    geom = feat.GetGeometryRef()
    #print(geom.GetGeometryName())
    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
    elif (geom.GetGeometryName() == 'POINT'):
        P_X=geom.GetPoint(0)[0]
        P_Y=geom.GetPoint(0)[1]
        xoff=int((P_X - xOrigin)/pixelWidth)
        yoff = round((yOrigin - P_Y)/pixelWidth)
        if xoff<0 or xoff>=xsize or yoff<0 or yoff>=ysize or P_X - xOrigin<0 or yOrigin - P_Y<0:
            return numpy.array([])
        else:
            Result=numpy.zeros((1,bands_num))
            num=0
            for R in Rasters:
                transform=R[0].GetGeoTransform()
                xOrigin = transform[0]
                yOrigin = transform[3]
                xoff=int((P_X - xOrigin)/pixelWidth)
                yoff = round((yOrigin - P_Y)/pixelWidth)
                for k in range(1,R[0].RasterCount+1):
                    banddataraster = R[0].GetRasterBand(k)
                    dataraster = banddataraster.ReadAsArray(xoff, yoff, 1, 1).astype(numpy.float32)
                    Result[0][num]=dataraster[0][0]
                    num=num+1
            return Result

    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")
    Points=[]
    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)
    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)+1
    yoff = int((yOrigin - ymax)/pixelWidth)+1
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount= int((ymax - ymin)/pixelWidth)+1
    c1=0
    c2=0
    c3=0
    c4=0
    if xcount==0 or ycount==0:
       return numpy.array([])
    if xoff+xcount<=0 or yoff+ycount<=0:
        return numpy.array([])
    if xoff<0:
        c1=-xoff
    if yoff<0:
        c2=-yoff
    if xoff+c1+xcount>=xsize:
        c3=xsize-(xoff+c1+xcount)
    if yoff+c2+ycount>=ysize:
        c4=ysize-(yoff+c2+ycount)
    # Read raster as arrays
    if xoff>=xsize:
        return numpy.array([])
    if yoff>=ysize:
        return numpy.array([])
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((xmin, pixelWidth, 0,ymax, 0, pixelHeight))
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1],options = ["ALL_TOUCHED=False"])
    bandmask = target_ds.GetRasterBand(1)
    target_ds2 = bandmask.ReadAsArray(c1, c2, xcount-c1+c3, ycount-c2+c4).astype(numpy.float)
    Result=numpy.zeros((int(sum(sum(target_ds2))),bands_num))
    #print(target_ds2)
    b_num=0
    for R in Rasters:
        x0_small=R[1]
        y0_small=R[2]
        bands=R[3]
        for k in range(1,bands+1):
            banddataraster = R[0].GetRasterBand(k)
            dataraster = banddataraster.ReadAsArray(xoff+R[1]+c1, yoff+R[2]+c2, xcount-c1+c3, ycount-c2+c4).astype(numpy.float32)
            num=0
            for i in range(target_ds2.shape[0]):
                for j in range(target_ds2.shape[1]):
                    if target_ds2[i][j]>0:
                        Result[num][b_num+k-1]=dataraster[i][j]
                        num=num+1
        b_num=b_num+bands
    return numpy.array(Result,dtype=numpy.float32)
def extract_data(tif_path,shp_path,field=[],tif_name=''):
    tif_param=Joint_Raster(tif_path,tif_name)
    Rasters=tif_param.Rasters
    X0_big=tif_param.X0
    Y0_big=tif_param.Y0
    xsize=tif_param.xsize
    ysize=tif_param.ysize
    bands_num=tif_param.bands_num
    pixel_width=tif_param.pixel_width
    pixel_height=tif_param.pixel_height
    #print(Rasters)
    
    projection = Rasters[0][0].GetProjection()
    src_shapefile = ogr.Open(shp_path, 1)
    if src_shapefile is None:
        sys.exit('ERROR: Open failed')
    layer = src_shapefile.GetLayer()
    layer_defn = layer.GetLayerDefn()
    # get shapefile attributes names list
    field_names = [layer_defn.GetFieldDefn(i).GetName()
                   for i in range(layer_defn.GetFieldCount())]
    fields=[]
    for i in field:
        if i in field_names:
            fields.append(i)
    result_pixels=numpy.empty((0,bands_num))
    if len(fields)==1:
        result_fields=numpy.empty((0))
    else:
        result_fields=numpy.empty((0,len(fields)))
    # get numpy data type from a gdal type
    A=enumerate(layer)
    for index, feature in enumerate(layer):
        sys.stdout.write('\r{} features processed'.format(index + 1))
        sys.stdout.flush()
        raster_block_filtered=zonal_stats(feature,shp_path,pixel_width,pixel_height,X0_big,Y0_big,Rasters,xsize,ysize,bands_num)
        if raster_block_filtered.shape[0]!=0:
            result_pixels=numpy.concatenate([result_pixels,raster_block_filtered],axis=0)
            ff=int(feature.GetField('CLASS'))
            result_fields=numpy.concatenate([result_fields,numpy.ones((raster_block_filtered.shape[0]))*ff],axis=0)
    src_shapefile = None
    sys.stdout.write('/n')
    if len(fields)!=0:
        return result_pixels,result_fields
    else:
        return result_pixels

def classify_bypath(path,i,tif_param):
    
    if i>=tif_param.ysize:
        return numpy.array([])
    
    Result=numpy.empty((tif_param.xsize,tif_param.bands_num),dtype=numpy.float32)
    b_num=0
    for R in tif_param.Rasters:
        for k in range(R[0].RasterCount):
            print(R[1],i+R[2])
            band=R[0].GetRasterBand(k+1).ReadAsArray(R[1],i+R[2],tif_param.xsize,1).astype(numpy.float32)
            Result[0:tif_param.xsize,b_num:b_num+1]=numpy.reshape(band,(tif_param.xsize,1))
            b_num=b_num+1
    return Result
    
def classify(path,i0,clf, name='',prob=0,tif_name=''):
    tif_param=Joint_Raster(path,tif_name)
    form = "GTiff"
    driver = gdal.GetDriverByName(form)
    raster_srs = osr.SpatialReference()
    if path.find('s3://')==-1:
        if len(name)==0:
            output = driver.Create(os.path.join(name),tif_param.xsize,tif_param.ysize,1+prob,gdal.GDT_Byte)
        else:
            output = driver.Create(os.path.join(name),tif_param.xsize,tif_param.ysize,1+prob,gdal.GDT_Byte)
    raster_srs.ImportFromWkt(tif_param.Rasters[0][0].GetProjectionRef())    
    output.SetProjection(raster_srs.ExportToWkt())
    output.SetGeoTransform((tif_param.X0,tif_param.pixel_width,0,tif_param.Y0,0,tif_param.pixel_height))
    
    for i in range(i0,tif_param.ysize):
#    for i in range(i0,i0+200):
        res=classify_bypath(path,i,tif_param)
        zero_mask = (res==0).all(axis=1)
        y = clf.predict(clf.scaler.transform(res))
        y = numpy.reshape(y, (1,y.shape[0]))
        y[:,zero_mask]=0
        output.GetRasterBand(1).WriteArray(y,0,i)
        output.FlushCache()
        sys.stdout.write('\r{} /'.format(i+1) + str(tif_param.ysize))
        sys.stdout.flush()
    output=None
    return 1
    

def train_pro(data_train, class_train, data_test, class_test):
    X_scaler = preprocessing.StandardScaler().fit(data_train)
    X_tr_transformed = X_scaler.transform(data_train)
    clf = MLPClassifier(hidden_layer_sizes = (100, 20), solver='lbfgs', alpha=1e-3, max_iter = 1000)
    clf = clf.fit(X_tr_transformed, class_train)
    clf.scaler = X_scaler
    y = clf.predict(X_tr_transformed)
    y = clf.predict(X_scaler.transform(data_test))
    
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_tr_transformed, class_train)
    clf.scaler = X_scaler
    y = clf.predict(X_tr_transformed)
    y = clf.predict(X_scaler.transform(data_test))
    
    clf = svm.SVC()
    clf = clf.fit(X_tr_transformed, class_train)
    clf.scaler = X_scaler
    y = clf.predict(X_tr_transformed)
    y = clf.predict(X_scaler.transform(data_test))
    
    tuned_parameters = [{'kernel': ['rbf','sigmoid'], 'gamma': ['auto', 1e-3, 1e-4],
                     'C': [1e-3, 1e-2, 1e-1, 0.2, 1, 5, 10, 100, 1000]}]
    
    clf = GridSearchCV(svm.SVC(probability = True), tuned_parameters, cv=5)
    clf = clf.fit(X_tr_transformed, class_train)
    clf.scaler = X_scaler
    y = clf.predict(X_tr_transformed)
    y = clf.predict(X_scaler.transform(data_test))
    return clf

def train(data_train, class_train):
    X_scaler = preprocessing.StandardScaler().fit(data_train)
    X_tr_transformed = X_scaler.transform(data_train)

    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_tr_transformed, class_train)
    clf.scaler = X_scaler
    return clf
    
def read_temp(file_name1, file_name2):
    data_tr = pickle.load(open(file_name1, 'br'))
    data_tt = pickle.load(open(file_name2, 'br'))
    data_train = data_tr[0]
    data_test = data_tt[0]
    class_train = data_tr[1]
    class_test = data_tt[1]
    return data_train, data_test, class_train, class_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', action='store', help='folder with images path')
    parser.add_argument('shp_path', action='store', help='shape path')
    parser.add_argument('out_path', action='store', help='tif output path')
    args = parser.parse_args()
    shp_path = os.path.normpath(args.shp_path)
    folder_path = os.path.normpath(args.folder_path)
    out_path = os.path.normpath(args.out_path)
    
    
    data_train, class_train = extract_data(folder_path, shp_path, field=['CLASS'], tif_name='.tif')
    clf = train(data_train, class_train)
    time_0 = time.time()
    classify(folder_path, 0, clf, tif_name='.tif',name = out_path)
