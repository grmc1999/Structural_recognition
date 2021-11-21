#!C:/Users/grmc1/Anaconda3/envs/INNOVATE/python.exe

import sys
import getopt
sys.path.append("D:\\Documentos\\INNOVATE\\GH\\proyectox\\Simetrias\\Utils")

from Utilities import *
from MF import *
from Visualization_utilities import *
from transformation import Transformation
from Signatures import Signature
from Pipelines import *

def get_arg(line):
    prSt=(np.array(line.split(','))).reshape(-1,1)
    Arg={}
    for pair in prSt:
        pair=pair[0].split("=")
        Arg[pair[0]]=pair[1]
    return Arg

def main(argv):
    try:
        opts,args=getopt.getopt(argv,"i:m:",["ifile=","mode="])
    except:
        sys.exit(2)

    for opt,arg in opts:
        if opt=="-i":
            txt=arg
        elif opt=="-m":
            method=arg

    txt=open(txt)
    #process=txt.readlines()
    processes=[i.split('\n')[0] for i in txt.readlines()]
    for process in processes:
        fargs=get_arg(process)
        if method=="Simetries":
            try:
                out=Detect_simetries(path=                  fargs["path"],
                                    visualization=          (fargs["visualization"]=="True"),
                                    geometry_type=          fargs["geometry_type"],
                                    voxel_down_sample=      float(fargs["voxel_down_sample"]),
                                    NN_for_signature_build= int(fargs["NN_for_signature_build"]),
                                    random_frac=            float(fargs["random_frac"]),
                                    filtered_SS=            (fargs["filtered_SS"]=="True"),
                                    Cluster_min_samples=    int(fargs["Cluster_min_samples"]),
                                    Cluster_xi=             float(fargs["Cluster_xi"]))
                print(out)
            except Exception as e: print(e)
        elif method=="Non_Linear_PCA":
            try:
                out=Detect_Tube_NonLinear_PCA(
                                    path=                   fargs["path"],
                                    visualization=          (fargs["visualization"]=="True"),
                                    geometry_type=          fargs["geometry_type"],
                                    voxel_down_sample=      float(fargs["voxel_down_sample"]),
                                    NN_for_signature_build= int(fargs["NN_for_signature_build"]),
                                    random_frac=            float(fargs["random_frac"]),
                                    filtered_SS=            (fargs["filtered_SS"]=="True"),
                                    bandwidth=              float(fargs["bandwidth"]),
                                    min_bin_freq=           int(fargs["min_bin_freq"]))
                print(out)
            except Exception as e: print(e)
        
if __name__ == '__main__':
    main(sys.argv[1:])