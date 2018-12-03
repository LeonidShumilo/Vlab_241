import os
import datetime
import numpy as np
import osr,gdal
import math
import sys
ndvi_band1=0
ndvi_band2=1
def read(path,year=[]):
    list_tifs=os.listdir(path)
    list_del=[]
    for i in list_tifs:
        if  os.path.split(i)[1].rfind(".tif")==-1 or os.path.split(i)[1].rfind(".tif")-len(os.path.split(i)[1])!=-4:
            list_del.append(i)
            continue
        if year!=[] and (i not in list_del):
            c_year=int(i[i.rfind('_')+1:-4])
            if c_year<year[0] or c_year>year[1]:
                list_del.append(i)
    for i in list_del:
        list_tifs.remove(i)
    list_tifs=sorted(list_tifs)
    return [os.path.join(path,i) for i in list_tifs]
def Joint_Raster(raster_list,mod=1):
    Rasters=[]
    X0BigRaster=0
    Y0BigRaster=0
    XEBigRaster=0
    YEBigRaster=0
    PixelWidth=0
    PixelHeight=0
    bandsnum=0
    all_xLast=[]
    all_x0=[]
    all_yLast=[]
    all_y0=[]
    raster_srs = osr.SpatialReference()
    for i in raster_list:
        a=gdal.Open(i)
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
        if y0_c+y_size>=R[4]:
            y_size=y_size-(y0_c+y_size-R[4])
        if x0_c+x_size>=R[3]:
            x_size=x_size-(x0_c+x_size-R[3])
    class tif_param:
       Rasters = Rasters2
       X0 = X0BigRaster
       Y0 = Y0BigRaster
       xsize = x_size
       ysize = y_size
       bands_num = bandsnum
       pixel_width = PixelWidth
       pixel_height = PixelHeight

    return tif_param
def result_classification(tif_path,tif_param,result,i,ker,output):
    if output==None:
        form = "GTiff"
        driver = gdal.GetDriverByName(form)
        raster_srs = osr.SpatialReference()
        print(tif_path)
        output = driver.Create(os.path.join(tif_path),tif_param.xsize,tif_param.ysize,1,gdal.GDT_Float32)
        raster_srs.ImportFromWkt(tif_param.Rasters[0][0].GetProjectionRef())
        output.SetProjection(raster_srs.ExportToWkt())
        output.SetGeoTransform((tif_param.X0,tif_param.pixel_width,0,tif_param.Y0,0,tif_param.pixel_height))
        output.GetRasterBand(1).WriteArray(result,0,i)
        output.FlushCache()
        return output
    else:
        output.GetRasterBand(1).WriteArray(result,0,i)
        output.FlushCache()
        return output


def extract_data_j(j,tif_param,ker=20,tif_name=''):
    Rasters=tif_param.Rasters
    Raster_minus_coords=np.array(Rasters)[:,1:3]
    X0_big=tif_param.X0
    Y0_big=tif_param.Y0
    xsize=tif_param.xsize
    ysize=tif_param.ysize
    bands_num=tif_param.bands_num
    pixel_width=tif_param.pixel_width
    pixel_height=tif_param.pixel_height
    Train_pixels=[]
    Class_pixels=[]
    k=0
    for R in range(len(Rasters)):
        min_ker=ker
        if tif_param.ysize-j-ker<0:
            min_ker=ker-(j+ker-tif_param.ysize)
        if R==0:
            res=np.zeros((min_ker,tif_param.xsize,bands_num))
            #print(res.shape)
        z_mask=[]
        for b in range(Rasters[R][0].RasterCount):
            res[:,:,k:k+1]=Rasters[R][0].GetRasterBand(b+1).ReadAsArray(Raster_minus_coords[R][0],j+Raster_minus_coords[R][1],tif_param.xsize,min_ker).astype(np.float32).reshape(min_ker,tif_param.xsize,1)
            if np.sum(res[:,:,k:k+1]<100)!=0 or z_mask!=[]:
                if z_mask==[]:
                    z_mask=res[:,:,k:k+1]<100
                    for b_back in range(b,-1,-1):
                        #print(b_back)
                        res[:,:,k-b_back:k-b_back+1][z_mask]=0
                else:
                    z_mask_2=res[:,:,k:k+1]<100
                    z_mask[z_mask_2]=True
                    for b_back in range(b,-1,-1):
                        #print(b_back)
                        res[:,:,k-b_back:k-b_back+1][z_mask]=0
            k=k+1
                        
    Train_pixels.append(res)
    return np.array(Train_pixels,dtype=np.float32)
def median_dates(tif_path,result_path,ker=1000):
    list_tifs=read(tif_path)
    tif_param=Joint_Raster(list_tifs)
    output=None
    ysize=tif_param.ysize
    xsize=tif_param.xsize
    n_b=int(tif_param.bands_num/len(tif_param.Rasters))
    for j in range(0,ysize,ker):
        vec=extract_data_j(j,tif_param,ker)[0]
        #print(np.unique(vec))
        NDVI=np.zeros((vec.shape[0],vec.shape[1],len(tif_param.Rasters)),dtype=np.float32)
        for i in range(0,vec.shape[2],n_b):
            #print(NDVI.shape)
            NDVI[:,:,int(i/n_b)]=(vec[:,:,int(i/n_b)*n_b+ndvi_band1]-vec[:,:,int(i/n_b)*n_b+ndvi_band2])/(vec[:,:,int(i/n_b)*n_b+ndvi_band2]+vec[:,:,int(i/n_b)*n_b+ndvi_band1])
        NDVI[np.isnan(NDVI)]=0
        result=np.max(NDVI,axis=2)
        output=result_classification(result_path,tif_param,result,j,ker,output)
def trend_index(result_path,classification_map,years):
    list_tifs=read(result_path,years)
    list_tifs.append(classification_map)
    tif_param=Joint_Raster(list_tifs)
    Rasters=tif_param.Rasters
    Raster_minus_coords=np.array(Rasters)[:,1:3]
    classification_raster=Rasters[-1]
    Rasters=Rasters[:-1]
    print(tif_param.xsize,tif_param.ysize)
    crops=classification_raster[0].GetRasterBand(1).ReadAsArray(Raster_minus_coords[-1][0],Raster_minus_coords[-1][1],tif_param.xsize,tif_param.ysize).astype(np.float32)
    crops=(crops<=9)*(crops>=2)+(crops>=15)
    trend=0
    for i in range(1,len(Rasters)):
        if i==1:
            trend=Rasters[i][0].GetRasterBand(1).ReadAsArray(Raster_minus_coords[i][0],Raster_minus_coords[i][1],tif_param.xsize,tif_param.ysize).astype(np.float32)-Rasters[i][0].GetRasterBand(1).ReadAsArray(Raster_minus_coords[i-1][0],Raster_minus_coords[i-1][1],tif_param.xsize,tif_param.ysize).astype(np.float32)
        else:
            trend=trend+Rasters[i][0].GetRasterBand(1).ReadAsArray(Raster_minus_coords[i][0],Raster_minus_coords[i][1],tif_param.xsize,tif_param.ysize).astype(np.float32)-Rasters[i-1][0].GetRasterBand(1).ReadAsArray(Raster_minus_coords[i-1][0],Raster_minus_coords[i-1][1],tif_param.xsize,tif_param.ysize).astype(np.float32)
    area=np.sum(crops)
    nq=int(math.fabs(tif_param.pixel_width*tif_param.pixel_height))
    pos_trend=np.sum(trend[crops]>=0)
    f=open(os.path.join(result_path,'indicator2.4.1_'+str(years[0])+'_'+str(years[1])+'.txt'),'w')
    txt='Total agricultural land area (squere km): '+str(float(area*nq)/1000000.0)
    txt=txt+'\n'+'Productive and sustainable agriculture land area (squere km): '+str(float(pos_trend*nq)/1000000.0)
    txt=txt+'\n'+'Not productive agriculture land area (squere km): '+str(float((area-pos_trend)*nq)/1000000.0)
    txt=txt+'\n'+'Proportion of agricultural area under productive and sustainable agriculture :'+str((float(pos_trend)/float(area))*100)
    f.write(txt)
    f.close()
    return area*nq,float(pos_trend)/float(area),(area-pos_trend)*nq
def index_calculating(first_year,last_year,teritory_name,result_path='',data_path='',classification_map=''):
    if classification_map=='':
        classification_map=os.path.join(os.path.realpath(''),'Data',teritory_name,'classification_map.tif')
    if result_path=='':
        result_path=os.path.join(os.path.realpath(''),'results',teritory_name)
    if not os.path.exists(result_path):
        os.mkdir(os.path.join(os.path.realpath(''),'results',teritory_name))
    if data_path=='':
        data_path=os.path.join(os.path.realpath(''),'Data')
    for year in range(first_year,last_year+1):
        if not os.path.exists(os.path.join(result_path,'NDVI_'+str(year)+'.tif')):
            median_dates(os.path.join(data_path,teritory_name,str(year)),os.path.join(result_path,'NDVI_'+str(year)+'.tif'))
    area,index,neg=trend_index(result_path,classification_map,[first_year,last_year])
    return index*100,
if __name__=="__main__":
    if len(sys.argv)==4:
        index=index_calculating(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
        print(index)
    elif len(sys.argv)==5:
        index=index_calculating(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4])
        print(index)
    elif len(sys.argv)==6:
        index=index_calculating(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5])
        print(index)
    elif len(sys.argv)==7:
        index=index_calculating(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
        print(index)
    else:
        print('Incorrect Input')
