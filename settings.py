
# The name of the Solr collection where Wikidata is indexed
SOLR_COLLECTION = 'kbpearl_official_1'

# The path to the language model, trained with "tapioca train-bow"
LANGUAGE_MODEL_PATH='../../related_work/Wikidata_alldata/latest-all.bow.pkl'
# The path to the pagerank Numpy vector, computed with "tapioca compute-pagerank"
PAGERANK_PATH='../../related_work/Wikidata_alldata/wikidata_graph.pgrank.npy'
# The path to the trained classifier, obtained from "tapioca train-classifier"
CLASSIFIER_PATH='data/my_classifier.pkl'
