import nltk, json, copy
import spacy, pprint
from spacy import displacy
from collections import Counter
import en_core_web_sm
from pprint import pprint
from Lemmatizer import Lemmatizer
from nltk.stem import WordNetLemmatizer

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
wnl = WordNetLemmatizer()

class InfoExtraction():
    def __init__(self, text, openie_result):
        self.doc = text
        self.openie_result = openie_result
        self.nlp = en_core_web_sm.load()
        '''Read and extract the related information from the text and openie_results'''
        #(self.triple_dict, self.sentence_list, self.triples) = self.extract_info_from_input()
        self.sentence_list = self.extract_info_from_input()

    def extract_info_from_input(self):
        doc = self.nlp(self.doc)
        sentence_list = [sent.text for sent in doc.sents]
        return sentence_list

    def dict_extend(self, main_dict, sub_dict):
        for sub_key in sub_dict.keys():
            if sub_key in main_dict.keys():
                if sub_dict[sub_key] not in main_dict[sub_key]:
                    main_dict[sub_key].append(sub_dict[sub_key])
            else:
                main_dict.update({sub_key:[sub_dict[sub_key]]})
        return main_dict

    def name_entity_reg_spacy(self, sentence):
        doc = self.nlp(sentence)
        temp_dict = {}
        for ent in doc.ents:
            #print(ent.text, ent.label_)
            temp_dict.update({ent.text:ent.label_})
        return temp_dict

    def name_entity_reg_nltk(self, sentence):
        """Find a list of noun phrases in the given sentence."""
        temp_dict = {}
        for sent in nltk.sent_tokenize(sentence):
            for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                if hasattr(chunk, 'label'):
                    # print(chunk.label(), ' '.join(c[0] for c in chunk))
                    temp_dict.update({' '.join(c[0] for c in chunk): chunk.label()})
        return temp_dict

    def improve_noun_dict(self, old_noun_dict):
        temp_noun_dict = {}
        for old_noun, types in old_noun_dict.items():
            ''' If it is a name with title such as "Mr Obama"'''
            new_noun = old_noun
            for title in PERSONAL_TITLE:
                if old_noun.find(title)!= -1:
                    new_noun = old_noun[old_noun.find(title)+len(title)+1:]
            temp_noun_dict.update({new_noun:types})
        new_noun_dict = {}
        candiate_list = temp_noun_dict.keys()
        for new_noun, types in temp_noun_dict.items():
            ''' If it is only a short name such as 'Obama' or 'Barack' '''
            if new_noun == 'Title':
                continue
            temp = new_noun
            for candiate_noun in candiate_list:
                if candiate_noun.find(temp)!=-1:
                    temp = candiate_noun
            new_noun_dict.update({temp:types})
        return new_noun_dict

    def remove_plura(self, phrase):
        temp_subject_list = []
        for item in phrase.split(' '):
            if item not in ["QUANT_S_1", "QUANT_O_1", "QUANT_R_1", "$"]:
                temp_subject_list.append(item)

        return ' '.join(temp_subject_list)

    def canonicalization(self, triple_list):
        print('@'*40)
        print('canonicalization!')
        new_triple_list = []
        temp_subject = {}
        for i in range(0, len(triple_list)):
            #print('*'*30)
            #print(sentence_list[i])
            #print(triple_list[i])
            triple_dict = {}
            for t in triple_list[i]:
                print('-'*30)
                new_t = []
                '''------------------subject-------------------'''
                t_content = t.split('|')
                subject = self.find_official_named_entity_mention_for_noun(t_content[0], self.official_named_entity_mention_dict)
                print(t_content[0])
                print(subject)
                if t_content[0] in PRONOUNS:
                    subject = temp_subject
                temp_subject = subject
                #print(subject.keys())
                if len(subject)==1:
                    final_subject = list(subject.keys())[0]
                else:
                    final_subject = {}
                new_t.append({self.remove_plura(t_content[0]):final_subject})

                '''------------------relation-------------------'''
                relation = self.remove_plura(self.transfer_relation_phrase_to_norm_pattern(t_content[1]))
                print(t_content[1])
                print(relation)
                new_t.append({t_content[1]:wnl.lemmatize(relation, 'v')})

                '''------------------object-------------------'''
                if t_content[2]!= '':
                    object = self.find_official_named_entity_mention_for_noun(t_content[2], self.official_named_entity_mention_dict)
                    print(t_content[2])
                    print(object)
                    if len(object)==1:
                        final_object = list(object.keys())[0]
                    else:
                        final_object = {}
                    new_t.append({self.remove_plura(t_content[2]): final_object})
                else:
                    new_t.append({'':{}})

                triple_dict.update({t:new_t})
            new_triple_list.append(triple_dict)
        print('@' * 40)
        return new_triple_list

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

    def find_official_named_entity_mention_for_noun(self,noun, official_named_entity_mention_dict):
        """Find a list of named entity mentions hidden in the nouns"""
        new_noun = noun
        noun_list = new_noun.split(' ')
        for title in PERSONAL_TITLE:
            if title in noun_list:
                print('hey')
                print(noun, title)
                temp_noun = noun[noun.find(title) + len(title) + 1:]
                if temp_noun == ' ' or len(temp_noun)== 0 :
                    '''For the cases that titles such as 'Professor' serve as a subject or object'''
                    continue
                else:
                    new_noun = temp_noun
        target_named_entity_mention = {}
        for mention in official_named_entity_mention_dict.keys():
            if mention.find(new_noun)!= -1:
                #print('!!!!!!!!!! '+mention+' '+new_noun)
                target_named_entity_mention.update({mention:official_named_entity_mention_dict[mention]})
        return target_named_entity_mention

    def side_info_extraction_from_doc(self):
        '''Sentence Tokenization'''
        print(self.sentence_list)

        '''Named Entity Recognition'''
        noun_tag_dict = {}
        noun_tag_dict = self.dict_extend(noun_tag_dict, self.name_entity_reg_nltk(self.doc))
        noun_tag_dict = self.dict_extend(noun_tag_dict, self.name_entity_reg_spacy(self.doc))
        #print(noun_tag_dict)
        self.official_named_entity_mention_dict = self.improve_noun_dict(noun_tag_dict)
        print(self.official_named_entity_mention_dict)

        '''Time Stamp'''
        time_list = []
        for keys, values in noun_tag_dict.items():
            if 'DATE' in values:
                time_list.append(keys)
        print(time_list)

        '''Canonicalization'''
        '''Step 1: improve the noun phrases (e.g.) Refer "Obama", "Mr Obama" and "President Obama" to "Barack Obama" in the article. '''
        '''The assumption is that one family name/first name in one article will only refers to one person.'''
        triple_list_with_candidate_named_entity_mentions = self.canonicalization(list(self.openie_result.values()))
        print(triple_list_with_candidate_named_entity_mentions)

        return (self.sentence_list, self.official_named_entity_mention_dict, time_list,triple_list_with_candidate_named_entity_mentions)




def main():
    text = "The Luftwaffe was the aerial warfare branch of the German Wehrmacht during World War II. Germany's military air arms during the First World War, the Luftstreitkräfte of the Army and the Marine-Fliegerabteilung of the Navy, had been disbanded in 1920 as a result of the terms of the Treaty of Versailles which stated that Germany was forbidden to have any air force. During the interwar period, German pilots were trained secretly in violation of the treaty at Lipetsk Air Base. With the rise of the Nazi Party and the repudiation of the Versailles Treaty, the Luftwaffe was established on 26 February 1935. The Luftwaffe's Condor Legion fought during the Spanish Civil War, the conflict became a testing ground for new doctrines and aircraft. As a result, the Luftwaffe grew to become one of the most sophisticated, technologically advanced and battle-experienced air forces in the world when war began in Europe in 1939. By the summer of 1939, the Luftwaffe had twenty-eight Geschwaders (wings). The Luftwaffe was instrumental in contributing to the German victories across Poland and Western Europe. During the Battle of Britain, however, despite causing severe damage to the RAF's infrastructure and British cities during the subsequent Blitz, it did not achieve victory. The Allied bombing campaigns from 1942 gradually destroyed the Luftwaffe's fighter arm. The Luftwaffe was also involved in operations over the Soviet Union, North Africa and Southern Europe. Despite its belated use of advanced turbojet and rocket propelled aircraft for the destruction of Allied bombers fleets, the Luftwaffe was overwhelmed by the Allies' superior numbers and improved tactics, and a lack of trained pilots and aviation fuel. A last-ditch effort to win air superiority was launched, during the closing stages of the Battle of the Bulge, in January 1945 failed. With rapidly dwindling supplies of petroleum, oil and lubricants after this campaign, and as part of the entire Wehrmacht military forces as a whole, the Luftwaffe ceased to be an effective fighting force and after the defeat of the Third Reich, the Luftwaffe was disbanded in 1946. The Luftwaffe had only two commanders-in-chief throughout its history: Hermann Göring and latterly Generalfeldmarschall Robert Ritter von Greim. The Luftwaffe was involved in war crimes and atrocities, including strafing civilian refugees and conducting human experiments, during its history."
    openie_result = {'0': ['Luftwaffe|was aerial warfare branch of|German Wehrmacht|(+,CT)|NONE\n', 'Luftwaffe|was aerial warfare branch during|World War II|(+,CT)|NONE\n', 'Luftwaffe|was|aerial warfare branch|(+,CT)|NONE\n'], '1': ['Germany|has military air arms during|First World War|(+,CT)|NONE\n', "Germany 's military air arms during First World War|is Luftstreitkräfte of|Army|(+,CT)|NONE\n", "Germany 's military air arms during First World War|is Luftstreitkräfte of Marine-Fliegerabteilung of|Navy|(+,CT)|NONE\n", "Germany 's military air arms during First World War|had been disbanded in 1920 as|result of terms of Treaty of Versailles|(+,CT)|NONE\n", "Germany 's military air arms during First World War|had been disbanded in|1920|(+,CT)|NONE\n", 'Germany|was forbidden to have|QUANT_O_1 air force|(+,CT)|NONE\n'], '2': ['German pilots|were trained During|interwar period|(+,CT)|NONE\n', 'German pilots|were trained in|violation of treaty at Lipetsk Air Base|(+,CT)|NONE\n', 'German pilots|were trained|secretly|(+,CT)|NONE\n'], '3': ['Luftwaffe|was established on 26 February 1935 With rise of|Nazi Party|(+,CT)|NONE\n', 'Luftwaffe|was established on 26 February 1935 With repudiation of|Versailles Treaty|(+,CT)|NONE\n', 'Luftwaffe|was established on|26 February 1935|(+,CT)|NONE\n'], '4': ['Luftwaffe|has|Condor Legion|(+,CT)|NONE\n', 'conflict|became testing ground for|new doctrines|(+,CT)|NONE\n', 'conflict|became testing ground for|aircraft|(+,CT)|NONE\n'], '5': ['Luftwaffe|grew to become QUANT_R_1 of most sophisticated As|result|(+,CT)|NONE\n', 'Luftwaffe|grew to become QUANT_O_1 of|most sophisticated|(+,CT)|NONE\n', 'QUANT_S_1 of most sophisticated|is advanced air forces in|world|(+,CT)|NONE\n', 'QUANT_S_1 of most sophisticated|is battle-experienced air forces in|world|(+,CT)|NONE\n', 'war|began in Europe technologically advanced air forces in|world|(+,CT)|NONE\n', 'war|began in Europe battle-experienced air forces in|world|(+,CT)|NONE\n', 'war|began in 1939 technologically advanced air forces in|world|(+,CT)|NONE\n', 'war|began in 1939 battle-experienced air forces in|world|(+,CT)|NONE\n', 'war|began advanced air forces in|world|(+,CT)|NONE\n', 'war|began battle-experienced air forces in|world|(+,CT)|NONE\n'], '6': ['Luftwaffe|had QUANT_R_1 Geschwaders By|the summer of 1939|(+,CT)|NONE\n', 'Luftwaffe|had|QUANT_O_1 Geschwaders|(+,CT)|NONE\n', 'QUANT_S_1 Geschwaders|is|wings|(+,CT)|NONE\n'], '7': ['Luftwaffe|was instrumental in|contributing to German victories across Poland|(+,CT)|NONE\n', 'Luftwaffe|was instrumental in|contributing to German victories across Western Europe|(+,CT)|NONE\n', 'Luftwaffe|was|instrumental|(+,CT)|NONE\n'], '8': ['RAF|has|infrastructure|(+,CT)|NONE\n', 'it|did achieve victory During Battle of|Britain|(-,CT)|NONE\n', 'it|did achieve|victory|(-,CT)|NONE\n', "it|did achieve victory despite|causing severe damage to RAF 's infrastructure during subsequent Blitz|(-,CT)|NONE\n", 'it|did achieve victory despite|causing severe damage to British cities during subsequent Blitz|(-,CT)|NONE\n'], '9': ["Allied bombing campaigns from 1942|destroyed|Luftwaffe 's fighter arm|(+,CT)|NONE\n", 'Luftwaffe|has|fighter arm|(+,CT)|NONE\n'], '10': ['Luftwaffe|was involved in operations over|Soviet Union|(+,CT)|NONE\n', 'Luftwaffe|was involved in operations over|North Africa|(+,CT)|NONE\n', 'Luftwaffe|was involved in operations over|Southern Europe|(+,CT)|NONE\n'], '11': ['its|has belated use of|advanced turbojet|(+,CT)|NONE\n', 'its|has belated use of|rocket|(+,CT)|NONE\n', 'belated use of advanced turbojet|propelled aircraft for destruction of|Allied bombers fleets|(+,CT)|NONE\n', 'belated use of rocket|propelled aircraft for destruction of|Allied bombers fleets|(+,CT)|NONE\n', 'belated use of advanced turbojet|propelled|aircraft|(+,CT)|NONE\n', 'belated use of rocket|propelled|aircraft|(+,CT)|NONE\n', "Luftwaffe|was overwhelmed by|Allies ' superior numbers|(+,CT)|NONE\n", 'Allies|has|superior numbers|(+,CT)|NONE\n'], '12': ['last-ditch effort|be win|air superiority|(+,CT)|NONE\n', 'last-ditch effort to win air superiority|was launched during closing stages of Battle of Bulge in|January 1945|(+,CT)|NONE\n', 'last-ditch effort to win air superiority|was launched during|closing stages of Battle of Bulge|(+,CT)|NONE\n'], '13': ['Luftwaffe|ceased to be effective fighting force With dwindling supplies of|petroleum|(+,CT)|NONE\n', 'Luftwaffe|ceased to be after defeat of Third Reich With dwindling supplies of|petroleum|(+,CT)|NONE\n', 'Luftwaffe|ceased to be effective fighting force With dwindling supplies of|oil|(+,CT)|NONE\n', 'Luftwaffe|ceased to be after defeat of Third Reich With dwindling supplies of|oil|(+,CT)|NONE\n', 'Luftwaffe|ceased to be effective fighting force With dwindling supplies of|lubricants|(+,CT)|NONE\n', 'Luftwaffe|ceased to be after defeat of Third Reich With dwindling supplies of|lubricants|(+,CT)|NONE\n', 'Luftwaffe|ceased to be effective fighting force after|campaign|(+,CT)|NONE\n', 'Luftwaffe|ceased to be after defeat of Third Reich after|campaign|(+,CT)|NONE\n', 'Luftwaffe|ceased to be effective fighting force as|part of entire Wehrmacht military forces as whole|(+,CT)|NONE\n', 'Luftwaffe|ceased to be after defeat of Third Reich as|part of entire Wehrmacht military forces as whole|(+,CT)|NONE\n', 'Luftwaffe|ceased to be|effective fighting force|(+,CT)|NONE\n', 'Luftwaffe|ceased to be after defeat of|Third Reich|(+,CT)|NONE\n'], '14': ['Luftwaffe|had QUANT_R_1 commanders-in-chief throughout|history|(+,CT)|NONE\n', 'Luftwaffe|had|QUANT_O_1 commanders-in-chief|(+,CT)|NONE\n', 'its|has|history|(+,CT)|NONE\n'], '15': ['Luftwaffe|was involved in war crimes including strafing civilian refugees during|history|(+,CT)|NONE\n', 'Luftwaffe|was involved in atrocities during|history|(+,CT)|NONE\n', 'Luftwaffe|was involved in war crimes including conducting human experiments during|history|(+,CT)|NONE\n', 'Luftwaffe|was involved in|war crimes including strafing civilian refugees|(+,CT)|NONE\n', 'Luftwaffe|was involved in|atrocities|(+,CT)|NONE\n', 'Luftwaffe|was involved in|war crimes including conducting human experiments|(+,CT)|NONE\n', 'its|has|history|(+,CT)|NONE\n']}
    target = {'0': ['Q2564009', 'P361', 'Q128781'], '4': ['Q2564009', 'P607', 'Q10859'], '13': ['Q2564009', 'P361', 'Q128781'], '14': ['Q77139', 'P410', 'Q7820253']}
    info = InfoExtraction(text, openie_result)
    print(info.transfer_relation_phrase_to_norm_pattern('Michael Jordan investigated Miller'))


if __name__ == '__main__':
    main()
