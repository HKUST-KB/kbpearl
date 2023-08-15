import sys, json, os
from Lemmatizer import Lemmatizer
from pprint import pprint



class ReadPATTYPatterns():
    def __init__(self, file):
        self.target_pattern_file = file
        self.target_relation_index_file = '../data/PATTY/'+self.target_pattern_file+'_relation_index_final.json'
        self.target_index_relation_file = '../data/PATTY/'+self.target_pattern_file+'_index_relation_final.json'


    def main(self):
        if os.path.isfile(self.target_relation_index_file) and os.path.isfile(self.target_index_relation_file):
            relation_index_patterns = self.read_from_json(self.target_relation_index_file)
            index_relation_patterns = self.read_from_json(self.target_index_relation_file)
        else:
            (relation_index_patterns, index_relation_patterns) = self.read_PATTY_relation('../data/PATTY/'+self.target_pattern_file+'-patterns.txt')
        return (relation_index_patterns, index_relation_patterns)


    def read_from_json(self,file):
        with open(file) as f:
            data = json.load(f)
        return data


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

    def read_PATTY_relation(self, filename):
        '''Read the synonymous patterns of relation phrases from files'''
        relation_index_dict = {}
        relation_content_dict = {}
        file = open(filename, "r")

        patterns = file.readlines()
        print('-' * 30 + 'Reading ' + filename + ' (' + str(len(patterns)) + ' in total)' + '-' * 30)
        for i in range(1, len(patterns)):
            line = patterns[i]
            line_content = line.split('\t')
            '''
            tags = re.findall(re.compile(r'\[\[(.*?)\]\]'), line_content[1])
            for tag in tags:
                if tag in different_pos_tag:
                    continue
                else:
                    different_pos_tag.append(tag)
            '''
            # if len(line_content[1].split('$')) > 2 and float(line_content[2]) >= 0.5:
            if len(line_content[1].split('$')) > 2:
                sys.stdout.write("\r" + (line_content[0]))
                sys.stdout.flush()
                synonymous_info = line_content[1].split('$')[:-1]

                synonymous_content = []
                for phrase in synonymous_info:
                    phrase_new = phrase.replace('[[pro]]', '[[prp]]')
                    synonymous_content.append(phrase_new[:phrase_new.find(';')])

                norm_synonymous_content = []
                for phrase in synonymous_content:
                    final_word = self.transfer_relation_phrase_to_norm_pattern(phrase)
                    norm_synonymous_content.append(final_word)
                # print(norm_synonymous_content)

                relation_content_dict.update({int(line_content[0]): norm_synonymous_content})
                for phrase in norm_synonymous_content:
                    if phrase not in relation_index_dict.keys():
                        index_list = []
                        index_list.append(int(line_content[0]))
                        relation_index_dict.update({phrase: index_list})
                    else:
                        index_list = relation_index_dict[phrase]
                        index_list.append(int(line_content[0]))
                        relation_index_dict[phrase] = index_list

        with open(self.target_relation_index_file, 'w') as fp_1:
            json.dump(relation_index_dict, fp_1)
        with open(self.target_index_relation_file, 'w') as fp_2:
            json.dump(relation_content_dict, fp_2)

        return (relation_index_dict, relation_content_dict)

if __name__ == '__main__':
    (relation_index_dict, relation_content_dict) = ReadPATTYPatterns('wikipedia').main()
    pprint(relation_index_dict)
    pprint(relation_content_dict)

