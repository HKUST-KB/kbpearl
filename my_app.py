from bottle import route, run, default_app, static_file, request, abort, response
import bottle
import sys
import json
import os, requests
from pynif import NIFCollection
import logging
import settings

from opentapioca.wikidatagraph import WikidataGraph
from opentapioca.languagemodel import BOWLanguageModel
from opentapioca.tagger import Tagger
from opentapioca.classifier import SimpleTagClassifier
from pprint import pprint

'''
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

tapioca_dir = os.path.dirname(__file__)

bow = BOWLanguageModel()
if settings.LANGUAGE_MODEL_PATH:
    bow.load(settings.LANGUAGE_MODEL_PATH)
graph = WikidataGraph()
if settings.PAGERANK_PATH:
    graph.load_pagerank(settings.PAGERANK_PATH)
tagger = None
classifier = None
if settings.SOLR_COLLECTION:
    tagger = Tagger(settings.SOLR_COLLECTION, bow, graph)
    classifier = SimpleTagClassifier(tagger)
    if settings.CLASSIFIER_PATH:
        classifier.load(settings.CLASSIFIER_PATH)
'''

def annotate_api(text):
    if not classifier:
        print('no classifier!')
        mentions = tagger.tag_and_rank(text)
    else:
        mentions = classifier.create_mentions(text)
        classifier.classify_mentions(mentions)

    result = [m.json() for m in mentions]
    result_dict = {}
    for content in result:
        mention = text[content['start']:content['end']]
        result_dict.update({mention: content['tags']})
    return result_dict

text = 'Michael Jordan and Kurt Miller studied artificial intelligence.'
#annotation_dict = annotate_api(text)
#pprint(annotation_dict)
id = 'id%3AQ11660'
shell = ''' curl -s --header "Content-Type: application/json" \
                                    --request POST \
                                    http://localhost:8983/solr/kbpearl_collection_test_2/query?q=%s
                                ''' % (id)
result_str = json.loads(os.popen(shell).read())
pprint(result_str)
print(result_str["response"]["docs"][0]["label"])

'''
r2 = requests.post('http://localhost:8983/solr/kbpearl_collection_test_2/query',
            params={'q':'id:Q11660',
             'wt':'json',
             'indent':'off',
            },
            headers ={'Content-Type':'text/plain'})
r2.raise_for_status()
resp2 = r2.json()
pprint(resp2)
'''

#curl -X POST -H 'Content-type: application/json' -d '{filter:"id%3AQ11660"}' http://localhost:8983/solr/kbpearl_collection_test_2/query?





'''
find_entity_flag = False
for content in annotation_dict:
    print('*'*20)
    candidate_entities_details = {}

    annotated_mention = text[content['start']:content['end']]
    print(annotated_mention)
    for candidate in content['tags']:
        print(candidate['id'])
        if candidate['nb_sitelinks'] == 0:
            nb_sitelinks = 1
        else:
            nb_sitelinks = int(candidate['nb_sitelinks'])
        candidate_entities_details.update({candidate['id']: nb_sitelinks})

    pprint(candidate_entities_details)
'''