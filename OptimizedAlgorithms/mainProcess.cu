#include <string>
#include <iostream>
#include "open3d/Open3D.h"
#include <stdio.h>

__global__
void outputFromGPU(){
    int i = threadIdx.x; // ID of thread 0-31 in this example
    printf("Hello world from GPU this is thread #%d\n",i);
}



int main(int argc, char ** argv) {

  // default values for command line options
  outputFromGPU<<<1,32>>>();

  char* path_file=NULL;

  // parse command line
  for (int i=1; i<argc; i++) {
    if (0 == strcmp(argv[i], "-i")) {
      i++;
      if (i<argc) path_file = argv[i];
    }
    /*else if (0 == strcmp(argv[i], "-dx")) {
      i++;
      if (i<argc) opt_dx = atof(argv[i]);
    }*/
  }

  std::cout<<path_file<<"\n";

  auto pcd = open3d::io::CreatePointCloudFromFile(path_file);

  std::cout<<pcd<<"\n";

  std::cout<<pcd->GetCenter()<<"\n";

  open3d::visualization::Visualizer visualizer;

  std::shared_ptr<open3d::geometry::PointCloud> pcl_ptr(new open3d::geometry::PointCloud);
  *pcl_ptr = *pcd;
  pcl_ptr->NormalizeNormals();

  visualizer.CreateVisualizerWindow("Open3D", 1600, 900);
  visualizer.AddGeometry(pcl_ptr);
  std::cout<<"RUN WIN"<<"\n";
  visualizer.Run();
  visualizer.DestroyVisualizerWindow();
  std::cout<<"HELLO"<<"\n";

  return 0;
}