imp-sim-cluster
=============

Authorship clustering using impostor similarity. This code can be used for both Authorship Verification (given two documents decide whether they are written by the same author or not) as well as Authorship Clustering (given a large set of documents by an unknown number of authors, cluster the documents into author clusters).

This is the code from the paper 
Efficient Authorship Clustering Using Impostor Similarity 
by Patrick Verga, James Allan, Brian Levine, and Marc Liberatore at UMass Amherst

Our method uses Locality Sensitive Hashing to more efficiently perform agglomerative clustering using Impostor Similarity

You can adapt the code under the 'experiments' package to fit your needs. The BlogExperiment.java class can be used with data at
http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm into a dir called './data/blog'. The Experiment.java class is a good entry point for using the code and is able to take many different command line parameters.

For questions you can email me at pat@cs.umass.edu
