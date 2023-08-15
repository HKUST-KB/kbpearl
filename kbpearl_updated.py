# encoding=utf8

from side_info_extraction import InfoExtraction
from semantic_graph_construction import Graph
'''
from wiki_data_download import EntityInfoExtractionFromWikidata
from minhash_construction import MinHashConstruction
from Lemmatizer import Lemmatizer
from minhash_forest import MyMinHashForest
from semantic_graph_construction import Vertex
from keyphrase_preprocessing import KeyPhrasePreprocessing
import spacy
nlp = spacy.load("en_core_web_sm")
'''
from pprint import pprint
import json, os, nltk, time, bottle, sys, logging, re
from pyjarowinkler import distance
from bottle import route, run, default_app, static_file, request, abort, response
from pynif import NIFCollection
import settings
from multiprocessing import Pool

from opentapioca.wikidatagraph import WikidataGraph
from opentapioca.languagemodel import BOWLanguageModel
from opentapioca.tagger import Tagger
from opentapioca.classifier import SimpleTagClassifier

from read_PATTY_patterns import ReadPATTYPatterns



PERSONAL_TITLE = ['Mrs','Miss','Mistress','Madam','Dame','Lady','Mr', 'Master', 'Esquire', 'Esq','Sir', 'Sire', 'Gentleman', 'Lord','Mx','Dr', 'Mrs.','Miss.','Mistress.','Madam.','Dame.','Lady.','Mr.', 'Master.', 'Esquire.', 'Esq.','Sir.', 'Sire.', 'Gentleman.', 'Lord.','Mx.','Dr.', 'Professor']
STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't", 'I', 'Me', 'My', 'Myself', 'We', 'Our', 'Ours', 'Ourselves', 'You',
                     "You're", "You've", "You'll", "You'd", 'Your', 'Yours', 'Yourself', 'Yourselves', 'He', 'Him',
                     'His', 'Himself', 'She', "She's", 'Her', 'Hers', 'Herself', 'It', "It's", 'Its', 'Itself', 'They',
                     'Them', 'Their', 'Theirs', 'Themselves', 'What', 'Which', 'Who', 'Whom', 'This', 'That', "That'll",
                     'These', 'Those', 'Am', 'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'Being', 'Have', 'Has', 'Had',
                     'Having', 'Do', 'Does', 'Did', 'Doing', 'A', 'An', 'The', 'And', 'But', 'If', 'Or', 'Because',
                     'As', 'Until', 'While', 'Of', 'At', 'By', 'For', 'With', 'About', 'Against', 'Between', 'Into',
                     'Through', 'During', 'Before', 'After', 'Above', 'Below', 'To', 'From', 'Up', 'Down', 'In', 'Out',
                     'On', 'Off', 'Over', 'Under', 'Again', 'Further', 'Then', 'Once', 'Here', 'There', 'When', 'Where',
                     'Why', 'How', 'All', 'Any', 'Both', 'Each', 'Few', 'More', 'Most', 'Other', 'Some', 'Such', 'No',
                     'Nor', 'Not', 'Only', 'Own', 'Same', 'So', 'Than', 'Too', 'Very', 'S', 'T', 'Can', 'Will', 'Just',
                     'Don', "Don't", 'Should', "Should've", 'Now', 'D', 'Ll', 'M', 'O', 'Re', 'Ve', 'Y', 'Ain', 'Aren',
                     "Aren't", 'Couldn', "Couldn't", 'Didn', "Didn't", 'Doesn', "Doesn't", 'Hadn', "Hadn't", 'Hasn',
                     "Hasn't", 'Haven', "Haven't", 'Isn', "Isn't", 'Ma', 'Mightn', "Mightn't", 'Mustn', "Mustn't",
                     'Needn', "Needn't", 'Shan', "Shan't", 'Shouldn', "Shouldn't", 'Wasn', "Wasn't", 'Weren', "Weren't",
                     'Won', "Won't", 'Wouldn', "Wouldn't"]
PRONOUNS = ['he', 'him','himself','his','He','Him','Himself', 'His','she','her','herself','She','Her','Herself', 'they', 'them', 'themselves', 'They', 'Them', 'Themselves','their', 'Their']
QUESTIONS = ['which','Which','how', 'How', 'where', 'Where', 'what', 'What', 'who', 'Who', 'when', 'When']
AUXILLIARYY_VERBS = ['be','do','have','was','were','been','does','did','done','had', 'has']

SOLR_COLLECTION = 'http://localhost:8983/solr/kbpearl_official_1'
COMMON_PREDICATES = ['P2302', 'P1687', 'P31', 'P361']
PATTY_PATTERN = 'nyt'
MAX_PREDICATE_PREDICATE_IDF = 6790.0
MAX_PREDICATE_OBJECT_IDF = 6790.0
MAX_ENTITY_PREDICATE_IDF = 10000.0
MAX_ENTITY_OBJECT_IDF = 10000.0
MAX_POOL_SIZE = 25
MAX_LINKING_SIZE = 6
MIN_EDGE_WEIGHT = 0.01

class KBPearl():
    def __init__(self):
        '''
        :param self.relation_index_dict: relation pattern - index id dictionary
        :param self.index_relation_dict: index id - relation pattern dictionary

        '''
        (self.relation_index_dict, self.relation_content_dict) = ReadPATTYPatterns(PATTY_PATTERN).main()
        self.entity_predicate_weight_dict = self.read_from_json('../data/keyphrases/entity_predicate_weight.json')
        self.entity_object_weight_dict = self.read_from_json('../data/keyphrases/entity_object_weight.json')
        self.predicate_prediacte_weight_dict = self.read_from_json('../data/keyphrases/prediacte_predicate_weight.json')
        self.predicate_object_weight_dict = self.read_from_json('../data/keyphrases/prediacte_object_weight.json')

    def read_from_json(self,file):
        '''Read entity from the json file which directly stores a dict'''
        print('Reading '+file+' ......')
        with open(file) as f:
            data = json.load(f)
        return data

    def read_from_file(self, file):
        '''Read entity from the json file, which stores multiple dicts'''
        data = {}
        for line in open(file, 'r'):
            data.update(json.loads(line))
        return data

    def get_openie_output(self,file):
        openie_output = self.read_from_file(file)
        #pprint(openie_output)
        return openie_output

    def obtain_details_from_wikidata(self,id):
        #print('**'*30)
        #print(id)
        id = 'id%3A'+str(id)
        shell = ''' curl -s --header "Content-Type: application/json" \
                                            --request POST \
                                            %s/query?q=%s
                                        ''' % (SOLR_COLLECTION,id)
        try:
            result_str = json.loads(os.popen(shell).read())
        except:
            return {}
        #pprint(result_str)
        return(result_str["response"]["docs"][0])

    def obtain_candidate_entities(self, noun):
        new_noun = re.sub(re.compile(' '),'*', noun)
        #print(new_noun)
        '''Constraint: should be entities that start with Q, sort by nb_sitelinks'''
        query_condition = 'aliases%3A' + new_noun + '%20%26%26%20id%3AQ*' + '&sort=nb_sitelinks%20desc'
        shell = ''' curl -s --header "Content-Type: application/json" \
                                                            --request POST \
                                                            "%s/query?q=%s"
                                                        ''' % (SOLR_COLLECTION, query_condition)
        try:
            result_str = json.loads(os.popen(shell).read())
        except:
            return []
        return result_str['response']['docs']



    def obtain_candidate_predicates(self, relation, question_word):
        if relation == '':
            return []
        if relation in AUXILLIARYY_VERBS:
            return []
        if question_word not in ['how','How','when','When','where','Where'] and relation.find(' ')==-1:
            new_relation = relation+'*'
        elif question_word in ['when', 'When'] and relation.find(' ')==-1:
            new_relation = relation+'*on'
        elif question_word in ['where', 'Where']and relation.find(' ')==-1:
            new_relation = relation+'*in'
        elif question_word in ['How', 'how'] and relation.find(' ')==-1:
            new_relation = relation+'*of'
        else:
            new_relation = re.sub(re.compile(' '),'*', relation)
        #print(new_relation)
        '''Constraint: should be entities that start with P, sort by nb_statements'''
        query_condition = 'aliases%3A' + new_relation + '%20%26%26%20id%3AP*' + '&sort=nb_statements%20desc'
        shell = ''' curl -s --header "Content-Type: application/json" \
                                                            --request POST \
                                                            "%s/query?q=%s"
                                                        ''' % (SOLR_COLLECTION, query_condition)
        #print(shell)
        try:
            result_str = json.loads(os.popen(shell).read())
        except:
            return []
        '''
        result_list = []
        
        if question_word in ['how','How','when','When','where','Where'] and len(result_str['response']['docs'])!=0:
            for p in result_str['response']['docs']:
                if question_word == 'how' or question_word == 'How':
                    for aliase in p['aliases']:
                        if aliase.find('cause')!=-1:
                            result_list.append(p)
                            break
                if question_word == 'when' or question_word == 'When':
                    for aliase in p['aliases']:
                        if aliase.find('date')!=-1:
                            result_list.append(p)
                            break
                if question_word == 'where' or question_word == 'Where':
                    for aliase in p['aliases']:
                        if aliase.find('place')!=-1 or aliase.find('location')!=-1:
                            result_list.append(p)
                            break
                else:
                    result_list = result_str['response']['docs']
        else:
            result_list = result_str['response']['docs']
        '''
        return result_str['response']['docs']

    def annotate_api(self, text):
        if not classifier:
            print('no classifier!')
            mentions = tagger.tag_and_rank(text)
        else:
            mentions = classifier.create_mentions(text)
            classifier.classify_mentions(mentions)

        result =  [m.json() for m in mentions]
        result_dict = {}
        for content in result:
            mention = text[content['start']:content['end']]
            result_dict.update({mention:content['tags']})
        return result_dict

    def search_weight(self, target_list, weight_dict, max_value):
        weight_list = []
        for id in target_list:
            if str(id) in weight_dict.keys():
                weight_list.append(weight_dict[str(id)])
            else:
                weight_list.append(max_value)
        return weight_list

    def obtain_weight_list(self, target_list, target_type, target):
        '''
        Obtain the weights of the predicates and objects from the json files
        :param target_list is the list that contain the items to be calculated
        :param target_type == 'entity' or 'predicate'
        :param target == 'edges' or 'predicates',
        '''
        weight_list = []
        if target_type == 'entity' and target == 'predicates':
            weight_dict = self.entity_predicate_weight_dict
            weight_list = self.search_weight(target_list, weight_dict, MAX_ENTITY_PREDICATE_IDF)
        if target_type == 'entity' and target == 'edges':
            weight_dict = self.entity_object_weight_dict
            weight_list = self.search_weight(target_list, weight_dict, MAX_ENTITY_OBJECT_IDF)
        if target_type == 'predicate' and target == 'predicates':
            weight_dict = self.predicate_prediacte_weight_dict
            weight_list = self.search_weight(target_list, weight_dict, MAX_PREDICATE_PREDICATE_IDF)
        if target_type == 'predicate' and target == 'edges':
            weight_dict = self.predicate_object_weight_dict
            weight_list = self.search_weight(target_list, weight_dict, MAX_PREDICATE_OBJECT_IDF)
        return weight_list

    def obtain_knowledge(self, item_id, target_type, detail_dict, target):
        '''
        Obtain the knowledge from the dict stored in Solr for the target item
        :param target_id, which is the target item, e.g., "Q5" or "P101"
        :param target_type == 'entity' or 'predicate'
        :param target == 'edges' or 'predicates',
        '''

        target_list = []
        '''And itself'''
        if target_type == 'entity':
            target_list.append(int(item_id[item_id.find('Q')+1:]))
        else:
            target_list.append(int(item_id[item_id.find('P') + 1:]))

        if target in detail_dict.keys():
            '''Directly obtain the ids of the predicates and objects'''
            target_list = detail_dict[target]

            '''Obtain the labels of the predicates and objects'''
            '''
            for id in target_list:
                if target == 'edges':
                    id = 'Q'+str(id)
                if target == 'predicates':
                    id = 'P'+str(id)
                target_detail = self.obtain_details_from_wikidata(id)
                target_detail_list.extend(target_detail['label'])
            '''
        return (target_list)

    def list_intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def list_union(self, lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    def pair_sim(self, compare_tuple):
        item1_type = compare_tuple[2]
        item2_type = compare_tuple[3]
        item1_dict = compare_tuple[4]
        item2_dict = compare_tuple[5]
        #print(compare_tuple)

        if item1_type == item2_type:
            '''If entity-entity edges or predicate-predicate edges'''
            #print('predicates')
            (item1_predicates) = self.obtain_knowledge(compare_tuple[0], item1_type, item1_dict, 'predicates')
            (item2_predicates) = self.obtain_knowledge(compare_tuple[1], item1_type, item2_dict,'predicates')
            intersection = (self.list_intersection(item1_predicates,item2_predicates))

            if len(intersection)==0:
                p_sim = 0
            else:
                union = (self.list_union(item1_predicates, item2_predicates))
                #print(intersection)
                #print(union)
                #print(self.obtain_weight_list(intersection, 'entity', 'predicates'))
                #print(self.obtain_weight_list(union, 'entity', 'predicates'))
                p_sim = sum(self.obtain_weight_list(intersection, item1_type, 'predicates'))/sum(self.obtain_weight_list(union, item1_type, 'predicates'))
            #print(sim)

            #print('objects')
            (item1_entities) = self.obtain_knowledge(compare_tuple[0], item1_type, item1_dict, 'edges')
            (item2_entities) = self.obtain_knowledge(compare_tuple[1], item1_type, item2_dict,'edges')
            intersection =(self.list_intersection(item1_entities, item2_entities))

            if len(intersection)==0:
                en_sim = 0
            else:
                union =(self.list_union(item1_entities, item2_entities))
                #print(intersection)
                #print(union)
                #print(self.obtain_weight_list(intersection, 'entity', 'edges'))
                #print(self.obtain_weight_list(union, 'entity', 'edges'))
                en_sim = sum(self.obtain_weight_list(intersection, item1_type, 'edges'))/sum(self.obtain_weight_list(union, item1_type, 'edges'))
            #print(sim)
            #sim = p_sim + en_sim
            sim = en_sim

        else:
            '''Otherwise, this is an entity-predicate edge'''
            (entity_predicates_list) = self.obtain_knowledge(compare_tuple[0], 'entity', item1_dict, 'predicates')
            predicate = compare_tuple[1][compare_tuple[1].find('P')+1:]
            if int(predicate) in entity_predicates_list:
                sim = self.obtain_weight_list([predicate], 'entity', 'predicates')[0]
                #print(sim)
            else:
                sim = 0

        return (compare_tuple[0], compare_tuple[1], sim)

    def overlap_coefficient(self, a,b, base):

        min_len = min(len(a),len(b))
        if min_len == 0:
            return 0+base
        else:
            return len(self.list_intersection(a,b))/float(min_len)+base

    def obtain_relation_predicate_edge(self, relation, predicate_detail, base):
        (self.relation_phrase_index_dict, self.relation_index_phrase_dict) = ReadPATTYPatterns('nyt').main()
        #pprint(self.relation_phrase_index_dict)
        #pprint(self.relation_index_phrase_dict)
        relation_syns = [relation]
        for phrase in self.relation_phrase_index_dict.keys():
            if relation in phrase:
                temp_index_list = self.relation_phrase_index_dict[phrase]
                for index in list(set(temp_index_list)):
                    relation_syns.extend(self.relation_index_phrase_dict[str(index)])
        relation_syns = list(set(relation_syns))
        final_relation_syns = []

        pattern = re.compile('\[\[.*?\]\]')
        for phrase in relation_syns:
            temp = re.sub(pattern,"",phrase)
            if temp[0] == ' ':
                temp = temp[1:]
            if temp[-1] == ' ':
                temp = temp[:-1]
            final_relation_syns.append(temp)
        #print(relation_syns)
        final_relation_syns = list(set(final_relation_syns))
        #print(final_relation_syns)

        if len(predicate_detail)!= 0  and 'aliases' in predicate_detail.keys():
            predicate_syns = predicate_detail["aliases"]
        else:
            return base
        #print(predicate_syns)
        return(self.overlap_coefficient(final_relation_syns, predicate_syns, base))

    def get_official_noun_phrases_and_relation_phrases(self, single_text, official_named_entity_mention_dict, time_list,
                                    triple_list_with_candidate_named_entity_mentions, reconstrcut_flag):
        noun_phrase = {}
        relation_phrase = {}
        for triple_info in triple_list_with_candidate_named_entity_mentions:
            for triple, triple_result in triple_info.items():
                '''
                print('*'*40)
                if list(triple_result[0].keys())[0] not in noun_phrase.keys():
                    noun_phrase.update(triple_result[0])
                if list(triple_result[2].keys())[0] not in noun_phrase.keys():
                    noun_phrase.update(triple_result[2])
                if list(triple_result[1].keys())[0] not in relation_phrase.keys():
                    relation_phrase.update(triple_result[1])
                print('*' * 40)
                '''
                #print('*' * 40)

                sub = list(triple_result[0].keys())[0]
                sub_backup = list(triple_result[0].values())[0]
                #print(sub)
                #print(sub_backup)
                if sub not in noun_phrase.keys() and sub not in QUESTIONS:
                    if single_text[-1:] == '?':
                        temp_list = sub.split(' ')
                        final_list = []
                        for w in temp_list:
                            if w not in QUESTIONS:
                                final_list.append(w)
                        noun_phrase.update({' '.join(final_list):sub_backup})
                    else:
                        noun_phrase.update({sub: sub_backup})

                obj = list(triple_result[2].keys())[0]
                obj_backup = list(triple_result[2].values())[0]
                if obj not in noun_phrase.keys() and obj not in QUESTIONS:
                    if single_text[-1:] == '?':
                        if obj in QUESTIONS:
                            continue
                        temp_list = obj.split(' ')
                        final_list = []
                        for w in temp_list:
                            if w not in QUESTIONS:
                                final_list.append(w)
                        noun_phrase.update({' '.join(final_list): obj_backup})
                    else:
                        noun_phrase.update({obj:obj_backup})

                re = list(triple_result[1].keys())[0]
                re_backup = list(triple_result[1].values())[0]
                if re not in relation_phrase.keys():
                    if re_backup not in AUXILLIARYY_VERBS:
                        temp_list = re_backup.split(' ')
                        new_relation = []
                        for item in temp_list:
                            if (item in AUXILLIARYY_VERBS) or (item.lower() in AUXILLIARYY_VERBS):
                                continue
                            else:
                                new_relation.append(item)
                        re_backup = ' '.join(new_relation)

                    relation_phrase.update({re.lower(): re_backup.lower()})

                #print('*' * 40)

        '''Append the mentions to the list of noun phrases as well... Note that currently the detected types of the mentions are not utlized'''
        for mention, type in official_named_entity_mention_dict.items():
            if mention not in noun_phrase.keys() and mention not in noun_phrase.values():
                if single_text[-1:]=='?' and mention in QUESTIONS:
                    continue
                noun_phrase.update({mention:{}})

        #noun_phrase = list(set(noun_phrase))
        #relation_phrase = list(set(relation_phrase))
        #print(noun_phrase)
        #print(relation_phrase)

        return(noun_phrase, relation_phrase)

    def add_predicate(self, single_text):
        external_predicates = {}
        time_keywords = ['When','when','What time', 'what time']
        for item in time_keywords:
            if single_text.find(item)!= -1:
                external_predicates.update({item:['P585']})
        return external_predicates

    def senmantic_graph_constrcution(self, single_text, time_list, noun_phrase, relation_phrase, official_named_entity_mention_dict):
        '''Construct the semantic graph'''
        g = Graph()
        #print(noun_phrase)
        #print(relation_phrase)

        '''Step 1. Find candidate entities for the set of noun phrases (also find the predicates of these entities)'''
        print('---------------------Step 1: process the nouns------------------------')
        noun_entity_dict = {}
        entity_content_dict = {}
        '''For each noun, we determine its candidate entities...'''
        for noun, noun_backup in noun_phrase.items():
            if single_text[:-1] == '?' and noun in QUESTIONS:
                continue
            if noun == '':
                continue
            if noun_backup == {}:
                noun_backup = ''
            #print(noun, noun_backup)
            g.add_vertex(noun)
            candidate_entities_details = {}

            if noun in time_list or noun_backup in time_list:
                print('time!')
                continue
            query_list = []
            if noun_backup != '':
                query_list = self.obtain_candidate_entities(noun_backup)[:MAX_LINKING_SIZE]
            if len(query_list) == 0 and noun not in PERSONAL_TITLE and noun not in STOP_WORDS and noun not in PRONOUNS:
                query_list = self.obtain_candidate_entities(noun)[:MAX_LINKING_SIZE]
                '''if still zero, remove the "."'''
                if len(query_list) == 0 and noun.find('.')!=-1:
                    query_list = self.obtain_candidate_entities(re.sub(re.compile('\.'), '', noun))[:MAX_LINKING_SIZE]
                if len(query_list) == 0 and noun_backup.find('.')!=-1:
                    query_list = self.obtain_candidate_entities(re.sub(re.compile('\.'), '', noun_backup))[:MAX_LINKING_SIZE]
                '''Remove the plura'''
                if len(query_list) == 0  and noun[-1:] == 's':
                    query_list = self.obtain_candidate_entities(noun[:-1])[:MAX_LINKING_SIZE]
                '''Use capital version?'''
                if len(query_list) == 0 and noun.istitle() == False:
                    #print(noun.title())
                    query_list = self.obtain_candidate_entities(noun.title())[:MAX_LINKING_SIZE]
                if len(query_list) == 0 and noun.islower():
                    query_list = self.obtain_candidate_entities(noun.upper())[:MAX_LINKING_SIZE]
            if len(query_list)!=0:
                #print(query_list)
                for candidate in query_list:
                    if 'desc' not in candidate.keys():
                        continue
                    if 'desc' in candidate.keys() and candidate['desc'].find('Wikimedia disambiguation page') != -1:
                        continue
                    if candidate['nb_sitelinks'] == 0:
                        nb_sitelinks = 1
                    else:
                        nb_sitelinks = candidate['nb_sitelinks'][0]
                    candidate_entities_details.update({candidate['id']: nb_sitelinks})
                    entity_content_dict.update({candidate['id']: candidate})
            else:
               continue
            #print(candidate_entities_details)
            '''Calculate the weight of the noun-entity edges, and add the entities to the graph'''
            temp_list = [] # this records the related entity nodes, e.g., [('Q41421', 'entity'), ('Q3308285', 'entity'), ...]
            total_pop_num = 0
            for key, value in candidate_entities_details.items():
                total_pop_num += value
            en_count = 0
            for entity in candidate_entities_details.keys():
                g.add_vertex(entity)
                if total_pop_num!= 0:
                    weight = candidate_entities_details[entity] / float(total_pop_num)
                else:
                    weight = 1.0
                #print(noun, entity, weight)
                g.add_edge(noun, entity, weight)
                temp_list.append(entity)
                en_count +=1
                if en_count>=MAX_LINKING_SIZE:
                    break

            noun_entity_dict.update({noun: temp_list})
        print(noun_entity_dict)
        '''Step 2. Find candidate predicates for the set of relation phrases (also find the predicates of these predicates)'''
        print('---------------------Step 2: process the relations------------------------')

        relation_predicate_dict = {}
        predicate_base_weight_dict = {}
        predicate_content_dict = {}
        for relation, relation_backup in relation_phrase.items():
            candidate_predicates_details = {}
            g.add_vertex(relation)
            query_list = []

            if single_text[-1:]=='?':
                question_produce = False
                for word in QUESTIONS:
                    if single_text.find(word)!=-1:
                        query_list = self.obtain_candidate_predicates(relation_backup, word)
                        question_produce = True
                if question_produce == False:
                    query_list = self.obtain_candidate_predicates(relation_backup, '')
            else:
                query_list = self.obtain_candidate_predicates(relation_backup, '')


            for p in query_list:
                candidate_predicates_details.update({p['id']: p['nb_statements'][0]})
                predicate_base_weight_dict.update({p['id']: p['nb_statements'][0]})
                predicate_content_dict.update({p['id']: p})

            '''Process the ``be" phrase'''
            if relation_backup == 'be':
                type_list = []
                for noun, type in official_named_entity_mention_dict.items():
                    type_list.extend(type)
                type_list = list(set(type_list))
                if single_text[-1:]=='?' and 'GPE' in type_list:
                    candidate_predicates_details.update({'P17': 34})
                    predicate_base_weight_dict.update({'P17': 34})
                    predicate_content_dict.update({'P17': {'id': 'P17', 'revid': 1030138595, 'label': ['country'], 'desc': "sovereign state of this item; don't use on humans", 'edges': [495, 1532, 27, 1336, 2596, 1001, 131, 2341, 2012, 205, 6256, 3024240, 1763527, 131, 20116958, 22158803, 25348518, 21502838, 5, 202444, 11879590, 12308941, 9430, 16521, 4167410, 4167836, 11266439, 1875621, 29048322, 15056995, 15056993, 31, 31, 21510865, 3624078, 1151405, 2577883, 15239622, 26830017, 148837, 836688, 1250464, 3895768, 28171280, 59281, 6256, 161243, 24334893, 170156, 3024240, 7275, 1145276, 43702, 182547, 21503252, 31, 1072012, 21510851, 580, 1319, 585, 1326, 582, 518, 805, 1310, 2241, 131, 710, 531, 1706, 459, 3831, 828, 1534, 2270034, 148, 51, 513, 837, 148, 32085240, 22158802], 'predicates': [1659, 1629, 1647, 2875, 1628, 3713, 3254, 2302, 1855, 3734, 3709, 1282], 'types': '{}', 'aliases': ['land', 'state', 'host country', 'sovereign state', 'country'], 'nb_statements': [34], 'nb_sitelinks': [0], '_version_': 1653312053068169218}})
                else:
                    candidate_predicates_details.update({'P31':32})
                    predicate_base_weight_dict.update({'P31':32})
                    predicate_content_dict.update({'P31':{}})

            '''Add the predicates to the graph'''
            temp_list = []  # this records the related predicate nodes
            for predicate in candidate_predicates_details.keys():
                g.add_vertex(predicate)
                temp_list.append(predicate)

            relation_predicate_dict.update({relation: temp_list})

        print(relation_predicate_dict)

        '''Relation-Predicate Edges'''
        for relation, predicate_list in relation_predicate_dict.items():
            for predicate in predicate_list:
                weight = self.obtain_relation_predicate_edge(relation, predicate_content_dict[predicate],
                                                             predicate_base_weight_dict[predicate] / 100)
                # print(relation, predicate,weight)
                g.add_edge(relation, predicate, weight)

        graph_final_edge_list = []
        '''Entity-Entity Edges'''
        ''' build an entity-noun dict, and entity-hash dict to record which pair of entity needs to be compared with each other. '''
        entity_noun_dict = {}
        for noun, entity_list in noun_entity_dict.items():
            for entity in entity_list:
                entity_noun_dict.update({entity: noun})
        # print(entity_noun_dict)
        candidate_entity_list = list(entity_noun_dict.keys())

        start_time = time.time()

        '''Compare the entities'''
        if len(noun_entity_dict.keys()) > 1:
            entity_compare_list = []
            for en1 in candidate_entity_list:
                for en2 in candidate_entity_list:
                    if entity_noun_dict[en1] != entity_noun_dict[en2] and (
                    en2, en1, 'entity', 'entity', entity_content_dict[en2],
                    entity_content_dict[en1]) not in entity_compare_list:
                        # print(en1,en2)
                        entity_compare_list.append(
                            (en1, en2, 'entity', 'entity', entity_content_dict[en1], entity_content_dict[en2]))
            if len(entity_compare_list)>0:
                pool = Pool(min(len(entity_compare_list), MAX_POOL_SIZE))
                en_results = pool.map(self.pair_sim, entity_compare_list)
                pool.close()
                for tuple in en_results:
                    if tuple[2]!=0 and tuple[2]>=MIN_EDGE_WEIGHT:
                        g.add_edge(tuple[0], tuple[1], tuple[2])
                        graph_final_edge_list.append((tuple[0],tuple[1],tuple[2]))
                        #print(tuple[0],tuple[1],tuple[2])


        '''Predicate-Predicate Edges'''
        ''' build a predicate-relation dict, and predicate-hash dict to record which pair of predicates needs to be compared with each other. '''
        predicate_relation_dict = {}
        for relation, predicate_list in relation_predicate_dict.items():
            for predicate in predicate_list:
                predicate_relation_dict.update({predicate: relation})
        candidate_predicate_list = list(predicate_relation_dict.keys())

        '''Compare the predicates'''
        #print(predicate_relation_dict)
        #print(predicate_content_dict)
        if len(relation_predicate_dict.keys()) > 1:
            predicate_compare_list = []
            for p1 in candidate_predicate_list:
                for p2 in candidate_predicate_list:
                    if predicate_relation_dict[p1] != predicate_relation_dict[p2] and (
                    p2, p1, 'predicate', 'predicate', predicate_content_dict[p2],
                    predicate_content_dict[p1]) not in predicate_compare_list:
                        # print(p1,p2)
                        predicate_compare_list.append(
                            (p1, p2, 'predicate', 'predicate', predicate_content_dict[p1], predicate_content_dict[p2]))
            if len(predicate_compare_list)>0:
                pool = Pool(min(len(predicate_compare_list), MAX_POOL_SIZE))
                #print(predicate_compare_list)
                p_results = pool.map(self.pair_sim, predicate_compare_list)
                pool.close()
                for tuple in p_results:
                    if tuple[2] != 0 and tuple[2]>MIN_EDGE_WEIGHT:
                        #print(tuple[0], tuple[1], tuple[2])
                        g.add_edge(tuple[0], tuple[1], tuple[2])
                        graph_final_edge_list.append((tuple[0], tuple[1], tuple[2]))

        '''Entity-Predicate Edges'''
        entity_predicate_compare_list = []
        for en in candidate_entity_list:
            for p in candidate_predicate_list:
                if (en, p, 'entity', 'predicate') not in entity_predicate_compare_list:
                    entity_predicate_compare_list.append(
                        (en, p, 'entity', 'predicate', entity_content_dict[en], predicate_content_dict[p]))
        if len(entity_predicate_compare_list) > 0:
            pool = Pool(min(len(entity_predicate_compare_list), MAX_POOL_SIZE))
            en_p_results = pool.map(self.pair_sim, entity_predicate_compare_list)
            pool.close()
            # pprint(en_p_results)
            entity_weight_dict = {}
            for tuple in en_p_results:
                if tuple[2] != 0:
                    if tuple[0] in entity_weight_dict.keys():
                        entity_weight_dict[tuple[0]].update({tuple[1]: tuple[2]})
                    else:
                        entity_weight_dict.update({tuple[0]: {tuple[1]: tuple[2]}})
            # print(entity_weight_dict)
            for entity, content in entity_weight_dict.items():
                total_weight = 0
                for predicate, weight in content.items():
                    total_weight += weight
                for predicate, weight in content.items():
                    #print(entity, predicate, weight / total_weight)
                    g.add_edge(entity, predicate, weight / total_weight/5.0)
                    graph_final_edge_list.append((entity, predicate, weight / total_weight/5.0))




        '''Step 3. Presenting our semantic graph'''
        full_node_dict = {}
        for v in g:
            #print('---')
            temp_dict = {}
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                #print('( %s , %s, %f)' % (vid, wid, v.get_weight(w)))
                temp_dict.update({str(wid): float(v.get_weight(w))})
            full_node_dict.update({str(v.get_id()): temp_dict})

        temp_vertices_weight_dict = {}
        for v in g:
            # print('g.vert_dict[%s]=%s' % (v.get_id(), g.vert_dict[v.get_id()]))
            temp_vertices_weight_dict.update({str(v.id): v.weighted_degree})
            # for id, weight in temp_vertices_weight_dict.items():
            # print(id)
            # print(weight)

        sorted_list = sorted(temp_vertices_weight_dict.items(), key=lambda x: x[1])
        #print(sorted_list)

        '''Step 4. Disambiguation Algorithm'''
        for node in sorted_list:

            '''noun phrases'''
            for noun in noun_entity_dict.keys():
                if node[0] in noun_entity_dict[noun]:  # if find this entity in the noun phrase
                    if len(noun_entity_dict[noun]) == 1:  # if this is the only one entity linked to the noun, continue
                        continue
                    else:  # otherwise, if this is not the only one entity linked to the noun, remove this entity and delect it from the noun_entity_dict[noun]
                        temp_list = []
                        for candidate_entity in noun_entity_dict[noun]:
                            if candidate_entity != node[0]:
                                temp_list.append(candidate_entity)
                        noun_entity_dict[noun] = temp_list

            '''relation phrases'''
            for relation in relation_predicate_dict.keys():
                if node[0] in relation_predicate_dict[relation]:
                    if len(relation_predicate_dict[relation]) == 1:
                        continue
                    else:
                        temp_list = []
                        for candidate_predicate in relation_predicate_dict[relation]:
                            if candidate_predicate != node[0]:
                                temp_list.append(candidate_predicate)
                        relation_predicate_dict[relation] = temp_list
        #rint(noun_entity_dict)
        #print(relation_predicate_dict)

        ''' Create a new entity for those who shares very few common things with the other selected nodes. '''
        #pprint(graph_final_edge_list)
        new_entity_record = []
        new_predicate_record = []
        all_candidate_nodes_list = []
        for noun, entity in noun_entity_dict.items():
            all_candidate_nodes_list.extend(entity)
        for relation, predicate in relation_predicate_dict.items():
            all_candidate_nodes_list.extend(predicate)
        all_candidate_nodes_list = list(set(all_candidate_nodes_list))
        for node in all_candidate_nodes_list:
            sum = 0.0
            for tuple in graph_final_edge_list:
                if tuple[0] == node and tuple[1] in all_candidate_nodes_list:
                    sum += tuple[2]
                if tuple[1] == node and tuple[0] in all_candidate_nodes_list:
                    sum += tuple[2]
            if sum < 0.005:
                if node.find('Q') != -1:
                    new_entity_record.append(node)
                if node.find('P') != -1:
                    new_predicate_record.append(node)
        #print(new_entity_record)
        #print(new_predicate_record)

        for noun,entity in noun_entity_dict.items():
            if entity in new_entity_record:
                noun_entity_dict[noun] = ['new entity']
        for relation, predicate in relation_predicate_dict.items():
            if predicate in new_predicate_record:
                relation_predicate_dict[relation] = ["new predicate"]

        #print(noun_entity_dict)
        #print(relation_predicate_dict)

        external_predicate = self.add_predicate(single_text)
        for key,value in external_predicate.items():
            relation_predicate_dict.update({key:value})
        elapsed_time = time.time() - start_time
        print('time!!!!!!!!!!!!')
        print(elapsed_time)
        return (noun_entity_dict, relation_predicate_dict)


    def result_record(self, id, noun_entity_dict,relation_predicate_dict, result_file):
        '''Record the results'''
        output_dict = {}
        output_dict.update({"entities":noun_entity_dict})
        output_dict.update({"relations":relation_predicate_dict})
        final = {id:output_dict}
        pprint(final)
        self.write_to_file(result_file, final)

    def write_to_file(self, filename, content):
        record_file = open(filename, 'a+')
        json.dump(content, record_file)
        record_file.write('\n')
        record_file.close()


    def my_kbpearl(self, id, single_text, open_ie_result, result_file):
        print('---------------------A new task starts-------------------------')
        '''Stage 1: Knowledge Extraction'''
        info = InfoExtraction(single_text, open_ie_result)
        (sentence_list, official_named_entity_mention_dict, time_list,
         triple_list_with_candidate_named_entity_mentions) = info.side_info_extraction_from_doc()

        if triple_list_with_candidate_named_entity_mentions == [{}]:
            #self.result_record(id, {}, {}, result_file)
            return




        '''Stage 2: Knowledge Population'''
        '''2-1-1. Annotate the text and extract potential mentions: '''
        #annotated_mention_dict = self.annotate_api(single_text)

        '''2-1-2. Semantic Graph Construction and 2-2. Graph Densification Algorithm'''
        reconstrcut_flag = True
        (noun_phrase, relation_phrase) = self.get_official_noun_phrases_and_relation_phrases(single_text, official_named_entity_mention_dict, time_list, triple_list_with_candidate_named_entity_mentions, reconstrcut_flag)
        '''find candidate entities for the set of noun phrases (also find the predicates of these entities)'''
        (noun_entity_dict, relation_predicate_dict) = self.senmantic_graph_constrcution(single_text, time_list, noun_phrase, relation_phrase, official_named_entity_mention_dict)

        '''2-3. Record the results'''
        self.result_record(id, noun_entity_dict, relation_predicate_dict, result_file)




    def main(self,dataset, openie_tool):
        gt = self.read_from_file('../../data-preprocessing/'+dataset+'_ground_truth_final.json')
        #pprint(gt)
        open_ie_result = self.get_openie_output('../data/openie_results/'+dataset+'_'+openie_tool+'.json')
        #pprint(open_ie_result)
        result_file = '../data/kbpearl_output_new/'+dataset+'_'+openie_tool+'.json'

        existing_id = []
        if os.path.exists(result_file):
            existing_results = self.read_from_file(result_file)
            existing_id.extend(existing_results.keys())


        count = 0
        for id, content in gt.items():
            #count+=1
            #if count>5:
            #    break


            
            print('*'*30)
            print(id)

            # if id in existing_id:
            #    print('existing!')
            #    continue
            if id != "2018-06-07-'How to Train Your Dragon,' Now With More Dragons and More Romance ":
                continue
            
            single_text = content['text']
            print(single_text)
            print(content['result'])
            print('-'*20)
            print(open_ie_result[id])
            try:
                self.my_kbpearl(id, single_text, open_ie_result[id], result_file)
            except:
                print(id+' fails...')
                record_file = open('bug_'+dataset+'_'+openie_tool+'.txt', 'a+')
                record_file.write(dataset+' '+openie_tool+' '+ id +'\n')
                record_file.close()







if __name__ == '__main__':
    pearl = KBPearl()
    dataset = 'nytimes'
    openie_tool = 'minieS'
    pearl.main(dataset, openie_tool)



    #text = 'The Arno is a river in the Tuscany region of Italy. It is the most important river of central Italy after the Tiber.'
    #openie_result = {"0":["Arno|is river in Tuscany region of|Italy|(+,CT)|NONE\n","Arno|is|river|(+,CT)|NONE\n"],"1":["It|is most important river of|central Italy|(+,CT)|NONE\n","It|is most important river after|Tiber|(+,CT)|NONE\n","It|is|most important river|(+,CT)|NONE\n"]}

    #pearl.my_kbpearl('Qtest',text, openie_result, 'test')


