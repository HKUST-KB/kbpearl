import json,os
from multiprocessing import Pool
from pprint import pprint
SOLR_COLLECTION = 'http://localhost:8983/solr/kbpearl_official_1'

class KeywordProcessor():
    def __init__(self):
        self.entity_predicate_weihgt_file = '../data/keyphrases/entity_predicate_weight.json'
        self.entity_object_weight_file = '../data/keyphrases/entity_object_weight.json'
        self.predicate_predicate_weight_file = '../data/keyphrases/prediacte_predicate_weight.json'
        self.predicate_object_weight_file = '../data/keyphrases/prediacte_object_weight.json'

    def read_from_json(self,file):
        with open(file) as f:
            data = json.load(f)
        return data

    def obatin_details_from_wikidata(self,condition):
        # print('**'*30)
        # print(id)
        shell = ''' curl -s --header "Content-Type: application/json" \
                    --request POST \
                    %s/select?q=%s
                    ''' % (SOLR_COLLECTION, condition)
        print(shell)
        result_list = json.loads(os.popen(shell).read())
        return (result_list["response"]["docs"])

    def obtain_keyphrase_dict(self, target_list, output1, output2):
        total_num = len(target_list)
        final_predicate_dict = {}
        final_object_dict = {}

        predicate_dict = {}
        object_dict = {}
        for item in target_list:
            print('-'*40)
            print(item['id'])
            if 'predicates' in item.keys():
                predicate_list = item['predicates']
                for predicate in predicate_list:
                    if predicate in predicate_dict.keys():
                        predicate_dict[predicate] += 1
                    else:
                        predicate_dict.update({predicate: 1})
            if 'edges' in item.keys():
                object_list = item['edges']
                for object in object_list:
                    if object in object_dict.keys():
                        object_dict[object]+=1
                    else:
                        object_dict.update({object:1})

        #predicate_dict = sorted(predicate_dict.items(),key=lambda x: x[1], reverse=True)
        #object_dict = sorted(object_dict.items(),key=lambda x: x[1], reverse=True)
        print(predicate_dict)
        print(object_dict)

        for key, value in predicate_dict.items():
            final_predicate_dict.update({int(key):(float(total_num)/value)})
        for key, value in object_dict.items():
            final_object_dict.update({int(key):(float(total_num)/value)})
        print(final_predicate_dict)
        print(final_object_dict)

        '''Record in the json files'''
        with open(output1, 'w') as fp_1:
            json.dump(final_predicate_dict, fp_1)
        with open(output2, 'w') as fp_2:
            json.dump(final_object_dict, fp_2)

        return final_predicate_dict, final_object_dict

    def max_value_in_dict(self, dict):
        max = 0.0
        for key,value in dict.items():
            if value>max:
                max = value
        return max

    def process_keyphrases_of_entities(self):
        if os.path.isfile(self.entity_predicate_weihgt_file) and os.path.isfile(self.entity_object_weight_file):
            print('weight files already exists!')
            entity_predicate_dict = self.read_from_json(self.entity_predicate_weihgt_file)
            entity_object_dict = self.read_from_json(self.entity_object_weight_file)
            print(self.max_value_in_dict(entity_predicate_dict))
            print(self.max_value_in_dict(entity_object_dict))
        else:
            entity_file_statements = '../data/keyphrases/Wikidata_top_1000_nb_statements_entities.json'
            entity_file_sitelinks = '../data/keyphrases/Wikidata_top_10000_nb_sitelinks_entities.json'
            #entity_list = self.read_from_json(entity_file_statements)["response"]["docs"]
            entity_list2 = self.read_from_json(entity_file_sitelinks)["response"]["docs"]
            #entity_list.extend(entity_list2)
            #print(entity_list)
            #print(len(entity_list))

            (entity_predicate_dict, entity_object_dict) = self.obtain_keyphrase_dict(entity_list2, self.entity_predicate_weihgt_file, self.entity_object_weight_file)
            print(sorted(entity_predicate_dict.items(),key=lambda x: x[1], reverse=True)[20])
            print(sorted(entity_object_dict.items(), key=lambda x: x[1], reverse=True)[20])

    def process_keyphrases_of_predicates(self):
        if os.path.isfile(self.predicate_predicate_weight_file) and os.path.isfile(self.predicate_object_weight_file):
            print('weight files already exists!')
            predicate_predicate_dict = self.read_from_json(self.predicate_predicate_weight_file)
            predicate_object_dict = self.read_from_json(self.predicate_object_weight_file)
            print(self.max_value_in_dict(predicate_predicate_dict))
            print(self.max_value_in_dict(predicate_object_dict))
        else:
            predicate_file_statements = '../data/keyphrases/Wikidata_top_10000_nb_statements_predicates.json'
            predicate_list = self.read_from_json(predicate_file_statements)["response"]["docs"]
            (predicate_predicate_dict, predicate_object_dict) = self.obtain_keyphrase_dict(predicate_list, self.predicate_predicate_weight_file, self.predicate_object_weight_file)
            print(sorted(predicate_predicate_dict.items(), key=lambda x: x[1], reverse=True)[20])
            print(sorted(predicate_object_dict.items(), key=lambda x: x[1], reverse=True)[20])


    def main(self):
        self.process_keyphrases_of_entities()
        self.process_keyphrases_of_predicates()
        return 0



if __name__ == '__main__':
    # Get the json files by http://localhost:8983/solr/kbpearl_official_1/select?q=id%3AQ*&rows=1000&sort=nb_statements%20desc
    processor = KeywordProcessor()
    processor.main()
