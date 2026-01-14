import numpy as np

def euclidean_distance(point_one,point_two):
    distance=(point_two[0]-point_one[0])**2+(point_two[1]-point_one[1])**2 #add forloop for more dimensions
    euclidean_distance=np.sqrt(distance)
    print("euclidean distance is ",euclidean_distance)

def multi_euclidean_distance(point_one,point_two):
    distance=0
    for i in range(len(point_one)):

        distance=distance+(point_two[i]-point_one[i])**2
    euclidean_distance=np.sqrt(distance)   
    print("euclidean distance is ",euclidean_distance)

def manhattan_distance(point_one,point_two):
    distance=0
    for i in range(len(point_one)):

        distance=distance+abs(point_two[i]-point_one[i])
     
    print("manhattan_distance is ",distance)


manhattan_distance((1,1,1),(2,2,2))
