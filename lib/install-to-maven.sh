#!/bin/bash

LIBPATH=`pwd`

cd $LIBPATH/..

mvn install:install-file -Dfile=${LIBPATH}/jstylo.jar -DgroupId=drexel -DartifactId=jstylo -Dversion=1.0 -Dpackaging=jar

mvn install:install-file -Dfile=${LIBPATH}/jgaap-5.2.0-lite.jar -DgroupId=jgaap -DartifactId=jgaap -Dversion=5.2.0-lite -Dpackaging=jar

