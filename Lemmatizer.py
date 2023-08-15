from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
nlp = spacy.load("en_core_web_sm")

class Lemmatizer():

    def __init__(self):
        self.postag_dict = {'JJ':'adj', 'JJR':'adj', 'JJS':'adj','CC':'con', 'DT':'det', 'CD':'num', 'PRP':'prp', 'PRP$':'prp', 'WP':'prp', 'WP$': 'prp', 'MD': 'mod', 'NN': 'prp', 'NNS': 'prp', 'NNP': 'prp', 'NNPS': 'prp'}


    def get_wordnet_pos(self,tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatizeSentence(self, sentence):
        tokens = word_tokenize(sentence)
        tagged_sent = pos_tag(tokens)

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            #print(tag)
            wordnet_pos = self.get_wordnet_pos(tag[1])
            if wordnet_pos == None:
                lemmas_sent.append(tag[0])
            else:
                lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        return lemmas_sent

    def lemmatizeRelationInSenetnce(self, relation, triple):

        sentence = triple[0] + ' ' + triple[1] + ' ' + triple[2]
        tokens = word_tokenize(sentence)
        tagged_sent = pos_tag(tokens)
        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            #print(tag)
            wordnet_pos = self.get_wordnet_pos(tag[1])
            if tag[1] in self.postag_dict.keys():
                lemmas_sent.append('[[' + self.postag_dict[tag[1]] + ']]')
            else:
                if wordnet_pos == None:
                    lemmas_sent.append(tag[0])
                else:
                    target = wnl.lemmatize(tag[0], pos=wordnet_pos)
                    lemmas_sent.append(target)
        relation_list = relation.split(' ')
        #print(relation_list)
        #print(triple)
        #print(lemmas_sent)

        start_index = len(triple[0].split(' '))
        end_index = len(lemmas_sent)-len(triple[2].split(' '))
        #print(start_index)
        #print(end_index)
        final_word = []
        for i in range(start_index, end_index):
            final_word.append(lemmas_sent[i])
        #print(final_word)
        return ' '.join(self.margeIdenticalElementsInAList(final_word))

    def margeIdenticalElementsInAList(self,temp_list):
        final_list = []
        for i in range(0,len(temp_list)):
            if temp_list[i] == temp_list[i-1] and temp_list[i].find('[[')!=-1:
                continue
            else:
                final_list.append(temp_list[i])
        return final_list

    def find_relation_in_long_phrases(self, relation_phrase):
        relation_list = relation_phrase.split(' ')
        final_list = []
        for relation in relation_list:
            doc = nlp(relation)
            for token in doc:
                if token.pos_ == 'VERB':
                    final_list.append(relation)
        return final_list



if __name__ == '__main__':
    lem = Lemmatizer()
    sentence = 'football was a family of team sports that involve, to varying degrees, kicking a ball to score a goal.'
    sentence = 'he eventually becomes you'

    #print (lem.lemmatizeSentence(sentence))

    relation = ' help'
    triple = ['russia', ' help', ' his campaign']
    #print (lem.lemmatizeRelationInSenetnce(relation, triple))
    print(lem.find_relation_in_long_phrases('supervise Miller and play with hime in'))

    #doc = nlp("supervise")
    # Coarse-grained part-of-speech tags
    #for token in doc:
    #    print(token.pos_)
