#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int main(int argc, char** argv){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("Linea 12.pcd",*cloud)==-1){
        PCL_ERROR ("No se pudo leer Linea 12 \n");
        return(-1);
    }
    std::cout<<"Cargado"
            <<(cloud->width)*(cloud->height)
            <<"puntos desde Linea 12 con los siguientes campos: "
            <<std::endl;
    for (const auto& point: *cloud){
        std::cout<<"  "<<point.x<<" "<<point.y<<" "<<point.z<<std::endl;
        return(0);
    }
}