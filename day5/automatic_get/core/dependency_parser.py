#!/usr/bin/env python
# coding: utf-8

import sys
import re
from pyltp import SentenceSplitter, Postagger, Parser, Segmentor, SementicRoleLabeller
# from transformers import BertTokenizer
# from transformers import BertForSequenceClassification
# from transformers import BertPreTrainedModel
# import torch
# import torch.nn as nn
# from transformers import BertModel
#
#
# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertForSequenceClassification, self).__init__(config)
#         self.num_labels = config.num_labels
#
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
#
#         self.init_weights()
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
#                 position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
#
#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask,
#                             inputs_embeds=inputs_embeds)
#
#         pooled_output = outputs[1]
#
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#
#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = nn.MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = nn.CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs
#
#         return outputs  # (loss), logits, (hidden_states), (attentions)
#
# tokenizer = BertTokenizer.from_pretrained('automatic_get/core/chinese-bert-wwm')
# model = BertForSequenceClassification.from_pretrained('automatic_get/core/chinese-bert-wwm',num_labels=3)
# model.load_state_dict(torch.load('models/wiki/model.pth'))


class Dependency(object):
    cws_model = "models/ltp_data_v3.4.0/cws.model"
    pos_model = "models/ltp_data_v3.4.0/pos.model"
    par_model = "models/ltp_data_v3.4.0/parser.model"
    ner_model = "models/ltp_data_v3.4.0/ner.model"
    # windows系统pisrl需要专门的模型
    pisrl_model = "models/ltp_data_v3.4.0/pisrl_win.model" if 'win' in sys.platform else "models/ltp_data_v3.4.0/pisrl.model"

    def __init__(self, sentence):
        self.sentence = sentence
        self.sentence_list = self.get_sentence_list()
        self.word_list = self.get_word_list()
        self.postag_list = self.get_postag_list()
        self.parser_list = self.get_parser_list()

    def get_sentence_list(self):
        """
        获取分句
        :return:
        """
        sents = SentenceSplitter.split(self.sentence)
        return [s for s in sents if s]

    def get_word_list(self):
        """
        获取分词
        :return:
        """
        segmentor = Segmentor()
        segmentor.load_with_lexicon(Dependency.cws_model, 'automatic_get/core/word.txt')
        # segmentor.load(Dependency.cws_model)
        try:
            word_list = [list(segmentor.segment(sentence)) for sentence in self.sentence_list]
        except Exception as e:
            word_list = [[]]
        finally:
            segmentor.release()
            return word_list

    def get_postag_list(self):
        """
        获取词性标注
        :param word_list:
        :return:
        """
        postag = Postagger()
        postag.load(Dependency.pos_model)
        try:
            postag_list = list(list(postag.postag(word_list)) for word_list in self.word_list)
        except Exception as e:
            postag_list = [[]]
        finally:
            postag.release()
            return postag_list

    def get_parser_list(self):
        """
        获取依存关系
        :param word_list:
        :param postag_list:
        :param model:
        :return:
        """
        parser = Parser()
        parser.load(Dependency.par_model)
        try:
            parser_list = [[(parser.head, parser.relation) for parser in parser.parse(w, p)] for w, p in
                           zip(self.word_list, self.postag_list)]
        except Exception as e:
            parser_list = [[]]
        finally:
            parser.release()
            return parser_list

    def get_role_list(self, words, postags):
        parser = Parser()
        parser.load(Dependency.par_model)

        rolelabel = SementicRoleLabeller()
        rolelabel.load(Dependency.pisrl_model)
        try:
            parsers = parser.parse(words, postags)
            roles = rolelabel.label(words, postags, parsers)
        except Exception as e:
            roles = [[]]
        finally:
            parser.release()
            rolelabel.release()
            return roles


def get_speech(text):
    text = re.sub('\\n;*|\\\\n|[\n\u3000\r]|（小标题）.*\\n|【.*】|（记者.*）|(.?月.+日电)', ' ', text)

    punc_set = set(['，', ',', '：', ':', ';', '；', '。', '!', '！', '？', '?'])
    nr_set = set(['ni', 'nh', 'n', 'r'])

    # 创建对象
    dependency_obj = Dependency(text)
    # 获取对象分句
    sentences = dependency_obj.sentence_list
    # 将对象分句分词
    words = dependency_obj.word_list
    # 获取对象分句中词性
    postags = dependency_obj.postag_list
    # 获取分句中依存
    parsers = dependency_obj.parser_list
    # 已经提前取好静态 “说”近义词
    similar_say = set(['说', '表示', '坦言', '指出', '告诉', '认为', '强调', '分析', '描述', '称', '预测', '爆料',
                       '指责', '否认',  '介绍', '透露', '谴责', '宣布', '呼吁','承认', '要求', '提醒', '提到', '表态',
                       '吐槽', '称赞', '批评', '回应', '回答'])

    all_said = []
    for i in range(len(sentences)):  # i为分句下标
        for j in range(len(words[i])):  # j为该分句内分词下标
            # 主谓关系找主语和谓语为‘说’相近的动词的句子，这里只考虑主语为特定实体（人名、地名、机构名）或者一般性名词，未考虑主语为代词的情况，待完善
            if parsers[i][j][1] == 'SBV' and postags[i][j] in nr_set:
                said = []
                pos = parsers[i][j][0] - 1

                # 谓语动词为‘说’近义词或者谓语动词下一个词与谓语动词为并列关系且为‘说’的同义词
                if words[i][pos] in similar_say:
                    say = words[i][pos]
                elif pos + 1 < len(words[i]) and \
                                parsers[i][pos + 1][1] == 'COO' and \
                                parsers[i][pos + 1][0] == (pos + 1) and \
                                words[i][pos + 1] in similar_say:
                    say = words[i][pos + 1]
                else:
                    continue
                if parsers[i][pos-1][1] == 'ADV':
                    say = words[i][pos-1] + say

                # 前面的言论
                # 之前句子内的言论
                if i != 0 and not set(sentences[i - 1]) & similar_say:
                    if sentences[i - 1][-1] in ['”', '，', ',', ';', '；']:
                        m = words[i - 1].index('”')
                        if '“' in words[i - 1]:
                            n = words[i - 1].index('“')
                            if m - n > 3:
                                said = said + words[i - 1][n + 1:m]
                        else:
                            for k in range(1, i - 1):
                                if '“' in words[i - 1 - k]:
                                    n = words[i - 1 - k].index('“')
                                    said = said + words[i - 1 - k][n + 1:] + sentences[i - k:i - 1] + words[i - 1][:m]
                                    break

                # 本句前面的言论
                if '”' in words[i][:pos - 1]:
                    m = words[i].index('”')
                    if '“' in words[i]:
                        n = words[i].index('“')
                        said = said + words[i][n + 1:m]
                    else:
                        for k in range(1, i):
                            if '“' in words[i - k]:
                                n = words[i - k].index('“')
                                said = said + words[i - k][n + 1:] + sentences[i - k + 1:i] + words[i][:m]
                                break

                # 后面的言论
                # 本句后面的言论
                start = pos + 2 if words[i][pos + 1] in punc_set else pos + 1
                said = said + words[i][start:]

                if i != len(sentences) - 1:
                    if '“' not in sentences[i] and said[-1] != '。':
                        for k in range(i + 1, len(sentences)):
                            if '。' in words[k]:
                                n = words[k].index('。')
                                said = said + sentences[i + 1:k] + words[k][:n]
                                break

                    if '“' in sentences[i] and '”' not in sentences[i]:
                        for k in range(i + 1, len(sentences)):
                            if '”' in words[k]:
                                n = words[k].index('”')
                                said = said + sentences[i + 1:k] + words[k][:n]
                                break

                # 之后句子内的言论
                if i < len(sentences) - 1 and not set(sentences[i + 1]) & similar_say:
                    # 这段注释勿删！！！
                    # m = words[i + 1].index('“')
                    #  因为sentences[i + 1][0] == '“'，所以m=0
                    # if '”' in words[i + 1]:
                    #     n = words[i + 1].index('”')
                    #     if n - m > 3:
                    #         said = said + words[i + 1][m + 1:n]
                    # else:
                    #     if i < len(sentences) - 2:
                    #         for k in range(i + 2, len(sentences)):
                    #             if '”' in words[k]:
                    #                 n = words[k].index('”')
                    #                 said = said + words[i + 1][m + 1:] + sentences[i + 2:k] + words[k][:n]
                    #                 break
                    if sentences[i + 1][0] == '“':
                        if '”' in words[i + 1]:
                            n = words[i + 1].index('”')
                            if n > 3:
                                said = said + words[i + 1][1:n]
                        else:
                            if i < len(sentences) - 2:
                                for k in range(i + 2, len(sentences)):
                                    if '”' in words[k]:
                                        n = words[k].index('”')
                                        said = said + words[i + 1][1:] + sentences[i + 2:k] + words[k][:n]
                                        break

                # 言论发表者，如果是代词替换成指代的人
                name = words[i][j]
                if parsers[i][j-1][1] == 'ATT':
                    name = words[i][j - 1] + name
                if postags[i][j] == 'r':
                    roles = dependency_obj.get_role_list(words[i-1], postags[i-1])
                    for role in roles:
                        for arg in role.arguments:
                            if arg.name == 'A0':
                                name = ''.join(words[i - 1][arg.range.start:arg.range.end + 1])
                                break
                        break

                flag = True
                for k in range(j, pos):
                    if postags[i][k] == 'wp' and words[i][k] not in ['－', '（', '）']:
                        flag = False
                if not flag:
                    continue

                # 情感分析
                # tokenizer_list = tokenizer.encode(''.join(said), add_special_tokens=False)
                # tokenizer_list.extend([0] * (100 - len(tokenizer_list))) if len(
                #     tokenizer_list) < 100 else tokenizer_list
                # tokenizer_list.append(102)
                # word_l = [101] + tokenizer_list
                #
                # input_ids = torch.tensor(word_l).unsqueeze(0)
                # labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
                # outputs = model(input_ids, labels=labels)
                # loss, logits = outputs[:2]
                # _, predicted = torch.max(logits.data, 1)
                # all_said.append([name, say, ''.join(said), predicted[0].item()])
                all_said.append([name, say, ''.join(said)])

    return all_said


# data_source = "../../data/export_sql_1558435/content.csv"
# data = pd.read_csv(data_source, encoding='utf-8')
# news = data["content"][:500]
# text = random.choice(news)
# print(text)
# all_said = get_speech(text)
# print(all_said)
# text = "不少旅客在入住酒店时都会遭遇“卡片骚扰”。这些从门缝偷偷塞进来的小卡片，上面往往发布色情信息，成为酒店业的“牛皮癣”，酒店和住客都深受其害。而酒店方在阻止“卡片党”过程中，甚至被暴力对待。本月在全季酒店，就连续发生两起暴力事件，员工在拦阻“卡片党”散发小广告时，遭到对方的暴力殴打。“卡片党”为何会如此猖獗？酒店黄色“小卡片”究竟该谁来管？记者在调查中发现，对于在酒店散发小卡片的行为，如何惩处尚存在着法律空白。对此，酒店业内人士呼吁，能够完善相关立法，解决困扰酒店的这一顽疾。酒店方店长保安都被打出血6月2日晚17点40分左右，松江全季酒店员工在监控室发现两名“卡片党”在7楼派发招嫖广告，立即用对讲机进行了通知。店长随即与员工一起前往7楼查看。刚出电梯口，即与这chagjue两个人相遇。店长要求他们立即交出剩余的非法广告，并称已报警。但其中一人非但不收敛，反而破口大骂！此时，店长发现他手中还有大量小广告，想上前收缴，作为他们违法行为的证据。这时，其中一人突然挥拳猛击店长脸部头部。客房阿姨见状拨打了110，警察很快赶到。最终扭获一白衣男子，另一人通过消防楼梯逃脱。让人惊讶的是，店长认出这两人在今年2月27日也在酒店散发小广告，被移交警方处理过。但时隔3个多月，他们又故伎重演。110接警民警将白衣男子带走，对其予以24小时拘留。6月5日晚，杭州全季酒店保安在大堂也发现了发小卡片的人员，上前阻止并报警。然而，这些“卡片党”被警方带走后，居然又来了几个人，径直找到酒店保安予以报复，带到后门监控盲区进行殴打，导致保安鼻子出血，多处淤青。“卡片党”打人之后扬长而去。"
# all_said = get_speech(text)
# print(all_said)
