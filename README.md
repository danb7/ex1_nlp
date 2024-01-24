# ex1_nlp
First assignment in NLP course: Distributional Similarity
by: Daniel Bazar 314708181 & Lior Krengel 315850594

## project files
### Plotting Code
word_vectors.py - the main script. produce the desirable output for each part of the assignment
### Additional Scripts
* utils.py - auxiliary functions for running the main script
* map_dict.py - definitions of the manually judgments of the correctness of the similarities for evaluting MAP in the MAP part
### other files
* report.pdf - the final report
* pca_plot.png - the saved figure from the Dimensionality Reduction part

How to run
----------
```
python .\word_vectors.py
```

the code produce the output for every part of the assignment:
* Generating lists of the most similar words
* Polysemous Words
* Synonyms and Antonyms
* The Effect of Different Corpora
* Plotting words in 2D - Dimensionality Reduction
* Word-similarities in Large Language Model
* Mean Average Precision (MAP) evaluation

Notes
----------
you can change every parameter (top-n, words, etc.) in the code
