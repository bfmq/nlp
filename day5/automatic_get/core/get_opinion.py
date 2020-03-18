#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install snownlp')


# In[1]:


import pandas as pd
import jieba
import numpy as np
import random
import re
import gensim
from snownlp import SnowNLP
from random import sample
from scipy.spatial.distance import cosine
from collections import defaultdict
from gensim.models import Word2Vec
from pyltp import Segmentor
from pyltp import  SentenceSplitter,NamedEntityRecognizer,Postagger,Parser,Segmentor,SementicRoleLabeller


# In[2]:


from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig
from transformers import BertPreTrainedModel
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW


class BertForSequenceClassification(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# In[7]:


def get_speech(text):
    tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm')
    model = BertForSequenceClassification.from_pretrained('chinese-bert-wwm',num_labels=3)
    
    text = re.sub('\\n;*|\\\\n|[\n\u3000\r]|（小标题）.*\\n|【.*】|（记者.*）', ' ', text)
    
    punc_set = ['，',',','：',':',';','；','。','!','！','？','?']
    nr = ['ni','nh','n','r']

    sentences = split_sentences(text)
    words = [get_word_list(sent) for sent in sentences]
    postags = [get_postag_list(w) for w in words]
    parsers = [get_parser_list(w,p) for w, p in zip(words, postags)]
    
    #取‘说’的近义词
    wl = ['说','分析','报道','描述','谴责','宣布','告诉','称','呼吁','承认']
    similar_say = [w for w in get_similar_words(wl) if w not in['目击者','消息人士','测算','看来','估算']]
    similar_say.extend(['要求','提醒'])
    
#     content_dict = []
    content = {}
    start = 0
    end = -1
    said = []
    name = ''
    
    flag = False
    
    for i in range(len(sentences)):
        if not sentences[i]:
            continue
        for j in range(len(words[i])):
            #主谓关系找主语和谓语为‘说’相近的动词的句子，这里只考虑主语为特定实体（人名、地名、机构名）或者一般性名词，未考虑主语为代词的情况，待完善
            if parsers[i][j][1]=='SBV' and postags[i][j] in nr:
                pos = parsers[i][j][0]-1

                #谓语动词为‘说’近义词或者谓语动词下一个词与谓语动词为并列关系且为‘说’的同义词
                if words[i][pos] in similar_say: 
                    say = words[i][pos]
                    flag = True
                if '说' in words[i][pos]:
                    say = '说'
                    flag = True
                if pos+1 < len(words[i]):
                    if (parsers[i][pos+1][1]=='COO' and parsers[i][pos+1][0]==(pos+1) and words[i][pos+1] in similar_say): 
                        say = words[i][pos+1]
                        flag = True

                if flag == True:
                    #前面的言论
                    #之前句子内的言论  
                    con = True
                    for s in similar_say:
                        if (i>0) and (s in sentences[i-1]):
                            con = False
                            break
                            
                    if con and i>0 and (sentences[i-1][-1] in ['”','，',',',';','；']):
                        m = words[i-1].index('”')
                        if ('“' in words[i-1]):
                            n = words[i-1].index('“')
                            if m-n >3:
                                said = said+words[i-1][n+1:m]
                        else:
                            for k in range(1,i-1):
                                if '“' in words[i-1-k]:
                                    n = words[i-1-k].index('“')
                                    said = said+words[i-1-k][n+1:]+sentences[i-k:i-1]+words[i-1][:m]
                                    break
                    
                    #本句前面的言论
                    if ('”' in words[i][:pos-1]):
                        m = words[i].index('”')
                        if ('“' in words[i]):
                            n = words[i].index('“')
                            said = said+words[i][n+1:m]
                        else:
                            for k in range(1,i):
                                if '“' in words[i-k]:
                                    n = words[i-k].index('“')
                                    said = said+words[i-k][n+1:]+sentences[i-k+1:i]+words[i][:m]
                                    break
                    #后面的言论                
                    #本句后面的言论
                    start = pos+1
                    if words[i][start] in punc_set:
                        start += 1
                    said = said+words[i][start:]
                
                    if(i<len(sentences)-1 ) and ('“' not in sentences[i]) and (said[-1] !='。'):
                        for k in range(i+1,len(sentences)):
                                if '。' in words[k]:
                                    n = words[k].index('。')
                                    said = said + sentences[i+1:k] + words[k][:n]
                                    break 
                             
                    if (i<len(sentences)-1 ) and ('“' in sentences[i]) and ('”' not in sentences[i]):
                        for k in range(i+1,len(sentences)):
                                if '”' in words[k]:
                                    n = words[k].index('”')
                                    said = said +sentences[i+1:k] + words[k][:n]
                                    break
                                       
                    #之后句子内的言论
                    con = True
                    for s in similar_say:
                        if (i<len(sentences)-1) and (s in sentences[i+1]):
                            con = False
                            break
                    
                    if con and (i<len(sentences)-1) and (sentences[i+1][0]=='“'):
                        m = words[i+1].index('“')
                        if ('”' in words[i+1]):
                            n = words[i+1].index('”')
                            if n-m >3: 
                                said = said+words[i+1][m+1:n]
                        else:
                            if i<len(sentences)-2:
                                for k in range(i+2,len(sentences)):
                                    if '”' in words[k]:
                                        n = words[k].index('”')
                                        said = said+words[i+1][m+1:] +sentences[i+2:k]+ words[k][:n]
                                        break    
                     
                    #言论发表者，如果是代词替换成指代的人
                    name = words[i][j]
                    if parsers[i][j-1][1]=='ATT':
                        name = words[i][j-1]+name
                    if postags[i][j] == 'r':
                        roles = get_role_list(words[i-1],postags[i-1])
                        for role in roles:
                            for arg in role.arguments:
                                if arg.name == 'A0':
                                    name = ''.join(words[i-1][arg.range.start:arg.range.end+1])
                                    break
                            break
                    
                    score = SnowNLP(''.join(said)).sentiments
                    
                    #情感分析
                    word_l = tokenizer.encode(''.join(said), add_special_tokens=False)
                    if len(word_l)<100:
                        while(len(word_l)!=100):
                            word_l.append(0)
                    word_l.append(102)
                    l = word_l
                    word_l = [101]
                    word_l.extend(l)

                    input_ids = torch.tensor(word_l).unsqueeze(0)

                    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1

                    model.load_state_dict(torch.load('model/model.pth'))
                    outputs = model(input_ids, labels=labels)
                    loss, logits = outputs[:2]
                    _,predicted=torch.max(logits.data,1)
                    if predicted[0] == 2:
                        print('负面消息')

                    print(name, ' ', say, ':', ''.join(said), ' score:',score)
                    print()
                    
                    '''
                    这里我没合，你根据你要的结构合就好了，
                    很简单，按照我的提取规则，基本上名字包含相同部分可以判断为同一个人
                    比如：村上春树 和 村上  视为同一人
                    再比如：霍希亚尔·兹巴里 和 兹巴里 视为同一人
                    '''
                    
                    
#                     if name in content.keys():# 同一人/机构/团体等的言论合并
#                         content[name] += [''.join(said)]
#                     else:
#                         content[name] = [''.join(said)]
                        
                    said = []
                    
                    flag = False
                    
#     if content:
#         return content
#     else:
#         print('无人发表言论！')


# In[8]:


data = pd.read_csv('data/sqlResult_1558435.csv', encoding='utf-8')


# In[9]:


data = data.fillna('')


# In[10]:


news =  sample(list(data['content']),300)


# In[11]:


get_speech('伊誓言肃清反政府武装\n　　　　张伟\n　　伊拉克外长霍希亚尔·兹巴里１１日说，伊拉克政府将与库尔德自治区政府加强合作，联手肃清占领伊北部摩苏尔的反政府武装。同一天，反政府武装占领了土耳其位于摩苏尔的领馆，绑架了一名领事和多名工作人员。\n　　约５０万伊拉克人已经逃离了伊拉克第二大城市摩苏尔，但伊拉克石油产业目前尚未受到波及。\n\n　　（小标题）联手肃清\n　　兹巴里在雅典出席欧盟－阿拉伯国家联盟会议期间说：“巴格达和库尔德自治区政府将加强合作，联手肃清这些外国作战人员。”但他并没有详细说明，伊拉克政府军将如何与库尔德自治区武装合作。\n　　兹巴里称，摩苏尔落入恐怖分子手中具有“戏剧性”，希望伊拉克领导人能团结起来，共同应对这一“严重、致命的威胁”。\n　　“要很快做出回应，”他说，“你不能让这些人长时间待在那里，构筑防御工事。”\n　　当前，反政府武装已经全部控制了摩苏尔所在尼尼微省，并攻占了东部基尔库克省和南部萨拉赫丁省部分地区。外界普遍认为，反政府武装来自逊尼派极端组织“伊拉克和黎凡特伊斯兰国”，但也可能包括其他武装组织的人员。\n　　尼尼微省省长阿特勒·努杰菲１１日表示，“摩苏尔有能力收复失地，清剿所有外来者……我们有一个恢复安全的计划”。\n　　目前，努杰菲在伊拉克北部库尔德地区埃尔比勒市避难。他还指责伊拉克安全部队高级指挥官政府提供有关摩苏尔的错误情报，要求对他们进行审判。\n\n　　（小标题）绑架领事\n　　伊拉克一名警官上校１１日对媒体说，反政府武装当天占领了在摩苏尔的土耳其领馆，绑架了一名领事和２４名工作人员。\n　　“‘伊拉克和黎凡特伊斯兰国’成员绑架了这名土耳其领事和他的２４名保镖与助手，”这名警察上校说。\n　　土耳其政府一名消息人士１１日向媒体证实了这一消息，但表示有４８名土耳其人被绑架，包括领事、３名儿童和数名土耳其特种部队士兵。\n　　这名消息人士称，土耳其方面已经与摩苏尔的武装组织直接联系，以确保这些人员的安全。\n　　媒体报道，“伊拉克和黎凡特伊斯兰国”武装人员１０日还绑架了２８名土耳其卡车司机。他们当时正往摩苏尔一座发电站运送柴油。\n\n　　（小标题）纷纷逃离\n　　摩苏尔当地一些居民１１日描述，反政府武装人员当天占据着当地政府机构和银行，一些人身穿军服。他们还用扩音器喊话，要求政府雇员回去上班。\n　　３０岁的店主阿布·艾哈迈德称说：“我从上周四就没有开门营业，因为担心安全……我将留在摩苏尔，不管怎么说，这是我的城市，现在局面平静了。”\n　　２５岁的大学生巴萨姆·默罕默德也表示要留在这座先前拥有２００万人口的城市。“但我担心自由，我尤其担心他们将对我们实行新的法律，”他说。\n　　有人选择坚守的同时，许多人为躲避战火已经纷纷逃离摩苏尔。\n　　总部设在瑞士日内瓦的国际移民组织１１日说，约５０万名伊拉克人已经逃离了摩苏尔。这一数字由该组织在当地的消息源估算。\n　　国际移民组织称，摩苏尔发生暴力冲突已经导致许多平民伤亡，由于医院处在交火区域，平民无法接近。一些清真寺已经改作诊所，救治伤员。\n　　国际移民组织说，由于车辆禁止进入摩苏尔市中心，许多人只能步行逃离，面临遭到迫击炮袭击的风险。\n　　由于当地一个主要的自来水厂被炸毁，摩苏尔西部居民区缺少饮用水。此外，摩苏尔许多家庭缺少食品。\n\n　　（小标题）石油无碍\n　　总部设在纽约的咨询公司欧亚集团１１日说，“伊拉克和黎凡特伊斯兰国”攻占摩苏尔对伊拉克的石油出口影响有限。\n　　欧亚集团中东和北非部门主任艾哈姆·卡迈勒说，“伊拉克和黎凡特伊斯兰”在占领摩苏尔后，可以获得资金、装备和兵员补充，战斗力会进一步增强。\n　　但卡迈勒认为，这并不会造成伊拉克其他省份的安全局势急剧恶化，从而影响到伊拉克的石油出口。\n　　伊拉克一名高级官员称，伊拉克石油产业尚未受影响，因为石油设施')


# In[17]:


text = random.choice(news)
print(text)
get_speech(text)


# In[27]:


SnowNLP('今天本来很开心去海边玩，结果下雨了，没办法继续待在海边，只能回家，出行计划泡汤了。').sentiments


# In[19]:


get_speech('新华社美国海湖庄园4月6日电（记者周效政　陈贽　朱东阳）当地时间6日，国家主席习近平在美国佛罗里达州海湖庄园同美国总统特朗普举行中美元首会晤。两国元首进行了深入、友好、长时间的会晤，双方高度评价中美关系取得的历史性进展，同意在新起点上推动中美关系取得更大发展，更好惠及两国人民和各国人民。\n下午5时许，习近平和夫人彭丽媛抵达海湖庄园，特朗普和夫人梅拉尼娅在停车处热情迎接。两国元首夫妇合影留念，亲切交谈。两国元首夫妇共同欣赏了特朗普外孙女和外孙演唱中文歌曲《茉莉花》并背诵《三字经》和唐诗。\n会晤中，习近平指出，一段时间以来，我同总统先生保持了密切联系，进行了多次通话和通信。我很高兴应总统先生邀请来美国举行这次会晤。我愿同总统先生就中美关系和重大国际及地区问题深入交换意见，达成更多共识，为新时期中美关系发展指明方向。\n习近平强调，中美两国关系好，不仅对两国和两国人民有利，对世界也有利。我们有一千条理由把中美关系搞好，没有一条理由把中美关系搞坏。中美关系正常化45年来，两国关系虽然历经风风雨雨，但得到了历史性进展，给两国人民带来巨大实际利益。中美关系今后45年如何发展？需要我们深思，也需要两国领导人作出政治决断，拿出历史担当。我愿同总统先生一道，在新起点上推动中美关系取得更大发展。\n习近平指出，合作是中美两国唯一正确的选择，我们两国完全能够成为很好的合作伙伴。下阶段双方要规划安排好两国高层交往。我欢迎总统先生年内对中国进行国事访问。双方可以继续通过各种方式保持密切联系。要充分用好新建立的外交安全对话、全面经济对话、执法及网络安全对话、社会和人文对话4个高级别对话合作机制。要做大合作蛋糕，制定重点合作清单，争取多些早期收获。推进双边投资协定谈判，推动双向贸易和投资健康发展，探讨开展基础设施建设、能源等领域务实合作。要妥善处理敏感问题，建设性管控分歧。双方要加强在重大国际和地区问题上的沟通和协调，共同推动有关地区热点问题妥善处理和解决，拓展在防扩散、打击跨国犯罪等全球性挑战上的合作，加强在联合国、二十国集团、亚太经合组织等多边机制内的沟通和协调，共同维护世界和平、稳定、繁荣。\n特朗普再次欢迎习近平主席到访海湖庄园。特朗普表示，美中两国作为世界大国责任重大。双方应该就重要问题保持沟通和协调，可以共同办成一些大事。我对此次美中元首会晤充满期待，希望同习近平主席建立良好的工作关系，推动美中关系取得更大发展。特朗普愉快接受了习近平主席发出的访华邀请，并期待尽快成行。\n中美两国元首还介绍了各自正在推进的内外优先议程，并就有关地区热点问题交换了意见。（完）\n')


# In[18]:


get_speech('新华社深圳５月２０日电（记者陈宇轩）由中国科学院深圳先进技术研究院与深圳市南山区政府共建的“粤港澳大湾区青少年创新科学教育基地”２０日在深圳成立，中国科学院将会为粤港澳三地的中小学科普教育提供更多科研资源支持。\n　记者采访了解到，该基地以日常教研为基础，通过“巡讲＋同步互动课堂＋师资培训＋考级评测”的方式，一方面开设博士课堂、机器人课程等科学教育课程，另一方面开展创新创业教育与研学活动，探索科学教育的新模式。\n　其中，“博士课堂”是该基地科学教育的主要内容之一。目前，来自中国科学院深圳先进技术研究院的１５名博士生，正结合自身科研领域研发相关科普课程。他们将通过科学小实验、卡通视频、科学小故事等教学方式，让“高大上”的科学知识变得新颖有趣。\n　“中科院有很多高端的科研人才，深圳的创新氛围很浓厚，参加这样的课程和活动能够让香港青少年接触到创新的前沿领域。”香港中文中学联会主席、香港培正中学校长谭日旭说。\n　据中国科学院深圳先进技术研究院院长助理毕亚雷介绍，目前，该基地由中国科学院深圳先进技术研究院旗下的中科创客学院、香港中文中学联会、澳门科学技术协进会三方共同负责日常管理运营。香港培正中学、澳门培正中学、中国科学院深圳先进技术研究院实验学校是该基地的首批参与学校。\n　“我们开放自己的实验室，开设创新创业课程，让更多科研资源转化为科普资源。在未来，我们计划让这些资源覆盖粤港澳三地超过１００家中小学校，促进粤港澳青少年科学教育水平的提高。”毕亚雷说。（完）')

