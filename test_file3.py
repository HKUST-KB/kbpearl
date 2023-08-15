from multiprocessing import Pool
import json, os, nltk, time, bottle, sys, logging, re
from pprint import pprint
SOLR_COLLECTION = 'http://localhost:8983/solr/kbpearl_official_1'
AUXILLIARYY_VERBS = ['be','do','have']
from Lemmatizer import Lemmatizer
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

COMMON_PREDICATES = ['P2302', 'P1687', 'P31', 'P361']
PATTY_PATTERN = 'nyt'
MAX_PREDICATE_PREDICATE_IDF = 6790.0
MAX_PREDICATE_OBJECT_IDF = 6790.0
MAX_ENTITY_PREDICATE_IDF = 10000.0
MAX_ENTITY_OBJECT_IDF = 10000.0
MAX_POOL_SIZE = 40
MAX_LINKING_SIZE = 10
MIN_EDGE_WEIGHT = 0.01

class TestCase():
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
        print(result_str)
        return(result_str["response"]["docs"][0])

    def obtain_candidate_entities(self, noun):
        new_noun = re.sub(re.compile(' '),'*', noun)
        '''Constraint: should be entities that start with Q, sort by nb_sitelinks'''
        query_condition = 'aliases%3A' + new_noun + '%20%26%26%20id%3AQ*' + '&sort=nb_sitelinks%20desc'
        shell = ''' curl -s --header "Content-Type: application/json" \
                                                            --request POST \
                                                            "%s/query?q=%s"
                                                        ''' % (SOLR_COLLECTION, query_condition)
        result_str = json.loads(os.popen(shell).read())
        return result_str['response']['docs']


    def obtain_candidate_predicates(self, relation, question_word):
        if relation in AUXILLIARYY_VERBS:
            return []
        if question_word not in ['how','How','when','When','where','Where'] and relation.find(' ')==-1:
            new_relation = relation+'*'
        else:
            new_relation = re.sub(re.compile(' '),'*', relation)
        '''Constraint: should be entities that start with Q, sort by nb_statements'''
        query_condition = 'aliases%3A' + new_relation + '%20%26%26%20id%3AP*' + '&sort=nb_statements%20desc'
        shell = ''' curl -s --header "Content-Type: application/json" \
                                                            --request POST \
                                                            "%s/query?q=%s"
                                                        ''' % (SOLR_COLLECTION, query_condition)
        result_str = json.loads(os.popen(shell).read())
        result_list = []
        if question_word:
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
        return result_list

    def get_official_noun_phrases_and_relation_phrases(self, official_named_entity_mention_dict,
                                    triple_list_with_candidate_named_entity_mentions):

        noun_phrase = {}
        relation_phrase = {}
        for triple_info in triple_list_with_candidate_named_entity_mentions:
            for triple, triple_result in triple_info.items():
                print('*'*40)
                if list(triple_result[0].keys())[0] not in noun_phrase.keys():
                    noun_phrase.update(triple_result[0])
                if list(triple_result[2].keys())[0] not in noun_phrase.keys() and list(triple_result[2].keys())[0]!= '':
                    noun_phrase.update(triple_result[2])
                re = list(triple_result[1].keys())[0]
                re_backup = list(triple_result[1].values())[0]
                if re not in relation_phrase.keys():
                    if re_backup not in AUXILLIARYY_VERBS:
                        temp_list = re_backup.split(' ')
                        new_relation = []
                        for item in temp_list:
                            if item not in AUXILLIARYY_VERBS:
                                new_relation.append(item)
                        re_backup = ' '.join(new_relation)
                    relation_phrase.update({re:re_backup})
                print('*' * 40)

        '''Append the mentions to the list of noun phrases as well... Note that currently the detected types of the mentions are not utlized'''
        for mention, type in official_named_entity_mention_dict.items():
            if mention not in noun_phrase.keys() and mention not in noun_phrase.values():
                noun_phrase.update({mention:{}})

        #noun_phrase = list(set(noun_phrase))
        #relation_phrase = list(set(relation_phrase))
        print(noun_phrase)
        print(relation_phrase)

        return(noun_phrase, relation_phrase)

    def transfer_relation_phrase_to_norm_pattern(self, phrase):
        word_list = phrase.split(' ')
        final_word_list = {}
        for count in range(0, len(word_list)):
            final_word_list.update({count: word_list[count]})
        # print(final_word_list)

        """Deal with the phrases where "[[...]]" have been deleted"""
        meaningful_phrase = ''
        for word in final_word_list.values():
            if word.find('[[') == -1 and word.find(']]') == -1:
                meaningful_phrase += word + ' '
        meaningful_phrase = meaningful_phrase[:-1]
        # print (meaningful_phrase)

        final_meaningful_phrase_list = Lemmatizer().lemmatizeSentence(meaningful_phrase)
        # print(final_meaningful_phrase_list)

        for count, word in final_word_list.items():
            if word.find('[[') == -1 and word.find(']]') == -1:
                continue
            else:
                final_meaningful_phrase_list.insert(count, word)
        # print(final_meaningful_phrase_list)
        final_word = ' '.join(final_meaningful_phrase_list)
        return final_word

    def single_noun_process(self, time_list, noun, noun_backup):
        if noun_backup == {}:
            noun_backup = ''
        print(noun, noun_backup)
        candidate_entities_details = {}
        ''' if this is a a 'Time'-type entity, deal with it first, only select the 'Time'-type entities as candidate entities '''
        if noun in time_list or noun_backup in time_list:
            print('time!')
            return {}
        query_list = []
        if noun_backup != '':
            query_list = self.obtain_candidate_entities(noun_backup)
        if len(query_list) == 0 and noun not in PERSONAL_TITLE and noun not in STOP_WORDS and noun not in PRONOUNS:
            query_list = self.obtain_candidate_entities(noun)
            '''if still zero, remove the "."'''
            if len(query_list) == 0 and noun.find('.') != -1:
                query_list = self.obtain_candidate_entities(re.sub(re.compile('\.'), '', noun))
            if len(query_list) == 0 and noun_backup.find('.') != -1:
                query_list = self.obtain_candidate_entities(re.sub(re.compile('\.'), '', noun_backup))[
                             :MAX_LINKING_SIZE]
            if len(query_list) == 0 and noun[-1:] == 's':
                query_list = self.obtain_candidate_entities(noun[:-1])
        if len(query_list) != 0:
            query_list = query_list[:MAX_LINKING_SIZE]
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
        return candidate_entities_details


if __name__ == '__main__':
    t = TestCase()
    #t.obtain_candidate_entities('artificial intelligence')
    #t.obtain_candidate_entities('Diana')
    #t.obtain_candidate_predicates('', 'when')
    #triple_list_with_candidate_named_entity_mentions = [{'Diana|is|princess|(+,CT)|NONE\n': [{'Diana': 'Diana'}, {'is': 'be'}, {'princess': {}}], 'Diana|did die||(+,CT)|NONE\n': [{'Diana': 'Diana'}, {'did die': 'do die'}, {'': 'Diana'}]}]
    #official_named_entity_mention_dict = {'Diana': ['PERSON']}
    #t.get_official_noun_phrases_and_relation_phrases(official_named_entity_mention_dict,triple_list_with_candidate_named_entity_mentions)
    #print(t.transfer_relation_phrase_to_norm_pattern("was born in"))
    #print(t.single_noun_process([],'Michael', "Michael Jordan"))
    t.obtain_details_from_wikidata('P31')