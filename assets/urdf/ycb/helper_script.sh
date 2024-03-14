#!/bin/bash

collision_suffix="textured_vhacd.obj"
urdf_suffix="google_16k/"
for entry in */
do 
  newvar="$entry$urdf_suffix"
  echo $entry
done
