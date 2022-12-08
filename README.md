# DNA-Sequencing-using-Machine-Learning
A machine learning model for classifying DNA sequencing of various species &amp; check their genes functioning in terms of similarities. 

A human genome has about 6 billion characters or letters. If you think the genome(the complete DNA sequence) is like a book, it is a book about 6 billion letters of “A”, “C”, “G” and “T”. Now, everyone has a unique genome. Nevertheless, scientists find most parts of the human genomes are alike to each other.

As a data-driven science, genomics extensively utilizes machine learning to capture dependencies in data and infer new biological hypotheses. Nonetheless, the ability to extract new insights from the exponentially increasing volume of genomics data requires more powerful machine learning models. By efficiently leveraging large data sets, deep learning has reconstructed fields such as computer vision and natural language processing. It has become the method of preference for many genomics modeling tasks, including predicting the influence of genetic variation on gene regulatory mechanisms such as DNA receptiveness and splicing.
So here, we will understand DNA structure and how machine learning can be used to work with DNA sequence data.

**Pre requisits:**

Biopython :is a collection of python modules that provide functions to deal with DNA, RNA & protein sequence.

**pip install biopython**

Squiggle : a software tool that automatically generates interactive web-based two-dimensional graphical representations of raw DNA sequences.

**pip install Squiggle**

DNA sequence data usually are contained in a file format called “fasta” format. Fasta format is simply a single line prefixed by the greater than symbol that contains annotations and another line that contains the sequence:

“AAGGTGAGTGAAATCTCAACACGAGTATGGTTCTGAGAGTAGCTCTGTAACTCTGAGG”

In this repository, we are building a classification model that is trained on the human DNA sequence and can predict a gene family based on the DNA sequence of the coding sequence. To test the model, we will use the DNA sequence of humans, dogs, and chimpanzees and compare the accuracies.
