import json,os,re
from multiprocessing import Pool
from pprint import pprint
from read_PATTY_patterns import ReadPATTYPatterns

SOLR_COLLECTION = 'http://localhost:8983/solr/kbpearl_official_1'
PATTY_PATTERN = 'nyt'
MAX_PREDICATE_PREDICATE_IDF = 6790.0
MAX_PREDICATE_OBJECT_IDF = 6790.0
MAX_ENTITY_PREDICATE_IDF = 10000.0
MAX_ENTITY_OBJECT_IDF = 10000.0
MAX_POOL_SIZE = 30

class Testcase():
    def __init__(self):
        self.entity_predicate_weight_dict = self.read_from_json('../data/keyphrases/entity_predicate_weight.json')
        self.entity_object_weight_dict = self.read_from_json('../data/keyphrases/entity_object_weight.json')
        self.predicate_prediacte_weight_dict = self.read_from_json('../data/keyphrases/prediacte_predicate_weight.json')
        self.predicate_object_weight_dict = self.read_from_json('../data/keyphrases/prediacte_object_weight.json')

    def read_from_json(self,file):
        '''Read entity from the json file which directly stores a dict'''
        with open(file) as f:
            data = json.load(f)
        return data

    def obtain_details_from_wikidata(self,id):
        #print('**'*30)
        #print(id)
        id = 'id%3A'+str(id)
        shell = ''' curl -s --header "Content-Type: application/json" \
                                            --request POST \
                                            %s/query?q=%s
                                        ''' % (SOLR_COLLECTION,id)

        result_str = json.loads(os.popen(shell).read())
        return(result_str["response"]["docs"][0])

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
                #print(self.obtain_weight_list(intersection, 'entity', 'edges'))
                #print(self.obtain_weight_list(union, 'entity', 'edges'))
                en_sim = sum(self.obtain_weight_list(intersection, item1_type, 'edges'))/sum(self.obtain_weight_list(union, item1_type, 'edges'))
            #print(sim)
            sim = p_sim + en_sim

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

    def overlap_coefficient(self, a,b):
        min_len = min(len(a),len(b))
        if min_len == 0:
            return 0
        else:
            return len(self.list_intersection(a,b))/float(min_len)

    def obtain_relation_predicate_edge(self, relation, predicate_detail):
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

        predicate_syns = predicate_detail["aliases"]
        #print(predicate_syns)
        return(self.overlap_coefficient(final_relation_syns, predicate_syns))


    def main(self):

        noun_entity_dict = {'Kurt Miller': ['Q6446912'], 'artificial intelligence': ['Q11660'], '2010': [], 'Michael Jordan': ['Q41421', 'Q3308285', 'Q65029442', 'Q6831719', 'Q27069141', 'Q975131', 'Q6831716', 'Q1928047', 'Q6831715']}
        relation_predicate_dict = {'study': ['P812', 'P101', 'P2578'], 'supervise': ['P185'], 'supervise Miller in': ['P185']}
        relation_predicate_dict = {}
        graph_final_edge_list = []

        '''Entity-Entity Edges'''
        ''' build an entity-noun dict, and entity-hash dict to record which pair of entity needs to be compared with each other. '''
        entity_noun_dict = {}
        for noun, entity_list in noun_entity_dict.items():
            for entity in entity_list:
                entity_noun_dict.update({entity: noun})
        # print(entity_noun_dict)
        candidate_entity_list = list(entity_noun_dict.keys())
        '''Record the details of the entities'''
        entity_content_dict = {}
        for entity in candidate_entity_list:
            entity_content_dict.update({entity: self.obtain_details_from_wikidata(entity)})
        if len(noun_entity_dict.keys())>1:
            '''Compare the entities'''
            entity_compare_list = []
            for en1 in candidate_entity_list:
                for en2 in candidate_entity_list:
                    if entity_noun_dict[en1] != entity_noun_dict[en2] and (en2, en1, 'entity', 'entity', entity_content_dict[en2],entity_content_dict[en1]) not in entity_compare_list:
                        #print(en1,en2)
                        entity_compare_list.append((en1,en2, 'entity', 'entity', entity_content_dict[en1], entity_content_dict[en2]))
            pool = Pool(min(len(entity_compare_list), MAX_POOL_SIZE))
            en_results = pool.map(self.pair_sim, entity_compare_list)
            pool.close()
            for tuple in en_results:
                if tuple[2]!=0:
                    graph_final_edge_list.append((tuple[0],tuple[1],tuple[2]))
                    print(tuple[0],tuple[1],tuple[2])

        '''Predicate-Predicate Edges'''
        ''' build a predicate-relation dict, and predicate-hash dict to record which pair of predicates needs to be compared with each other. '''
        predicate_relation_dict = {}
        for relation, predicate_list in relation_predicate_dict.items():
            for predicate in predicate_list:
                predicate_relation_dict.update({predicate: relation})
        candidate_predicate_list = list(predicate_relation_dict.keys())
        '''Record the details of the predicates'''
        predicate_content_dict = {}
        for predicate in candidate_predicate_list:
            predicate_content_dict.update({predicate: self.obtain_details_from_wikidata(predicate)})
        if len(relation_predicate_dict.keys())>1:
            '''Compare the predicates'''
            predicate_compare_list = []
            for p1 in candidate_predicate_list:
                for p2 in candidate_predicate_list:
                    if predicate_relation_dict[p1]!= predicate_relation_dict[p2] and (p2,p1, 'predicate', 'predicate',predicate_content_dict[p2],predicate_content_dict[p1]) not in predicate_compare_list:
                        #print(p1,p2)
                        predicate_compare_list.append((p1,p2, 'predicate', 'predicate', predicate_content_dict[p1], predicate_content_dict[p2]))
            pool = Pool(min(len(predicate_compare_list), MAX_POOL_SIZE))
            p_results = pool.map(self.pair_sim, predicate_compare_list)
            pool.close()
            for tuple in p_results:
                if tuple[2]!=0:
                    graph_final_edge_list.append((tuple[0],tuple[1],tuple[2]))
                    print(tuple[0],tuple[1],tuple[2])

        '''Entity-Predicate Edges'''
        entity_predicate_compare_list = []
        for en in candidate_entity_list:
            for p in candidate_predicate_list:
                if (en, p, 'entity', 'predicate') not in entity_predicate_compare_list:
                    entity_predicate_compare_list.append((en, p, 'entity', 'predicate', entity_content_dict[en], predicate_content_dict[p]))
        if len(entity_predicate_compare_list)>0:
            pool = Pool(min(len(entity_predicate_compare_list), MAX_POOL_SIZE))
            en_p_results = pool.map(self.pair_sim, entity_predicate_compare_list)
            pool.close()
            #pprint(en_p_results)
            entity_weight_dict ={}
            for tuple in en_p_results:
                if tuple[2] != 0:
                    if tuple[0] in entity_weight_dict.keys():
                        entity_weight_dict[tuple[0]].update({tuple[1]: tuple[2]})
                    else:
                        entity_weight_dict.update({tuple[0]: {tuple[1]: tuple[2]}})
            #print(entity_weight_dict)
            for entity, content in entity_weight_dict.items():
                total_weight = 0
                for predicate, weight in content.items():
                    total_weight += weight
                for predicate, weight in content.items():
                    print(entity, predicate, weight / total_weight)
                    graph_final_edge_list.append((entity, predicate, weight / total_weight))

        '''Relation-Predicate Edges'''
        for relation, predicate_list in relation_predicate_dict.items():
            for predicate in predicate_list:
                print(relation, predicate, self.obtain_relation_predicate_edge(relation, predicate_content_dict[predicate]))

        pprint(graph_final_edge_list)

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
                if tuple[0] ==node and tuple[1] in all_candidate_nodes_list:
                    sum += tuple[2]
                if tuple[1] == node and tuple[0] in all_candidate_nodes_list:
                    sum += tuple[2]
            print(node)
            print(sum)
            if sum < 0.005:
                print('new!')
                if node.find('Q')!=-1:
                    new_entity_record.append(node)
                if node.find('P')!= -1:
                    new_predicate_record.append(node)
        print(new_entity_record)
        print(new_predicate_record)





if __name__ == '__main__':
    test = Testcase()
    test.main()

    #en_p_results = [('Q3308285', 'P101', 38.31417624521073),('Q3308285', 'P185', 192.30769230769232),('Q3308285', 'Pxx', 0),('Q4', 'P185', 11),('Q4', 'Pxx', 192)]


