# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import os

def process_single_domain_data():
    input_file = './original_data/data_full.json'
    domain_file = './domain_intent_map.json'

    jsn = json.load(open(input_file, 'r'))
    dmap_jsn = json.load(open(domain_file, 'r'))

    dmap = {}
    for domain in dmap_jsn:
        for intent in dmap_jsn[domain]:
            dmap[intent] = domain

    train = {}
    dev = {}
    test = {}
            
    for data_type in jsn:
        if 'train' in data_type:
            map = train
        elif 'val' in data_type:
            map = dev
        elif 'test' in data_type:
            map = test
        else:
            assert False
            
        for data in jsn[data_type]:
            if data[1] not in dmap:
                continue
            
            domain = dmap[data[1]]

            if domain not in map:
                map[domain] = []
            
            map[domain].append(data)

    for map, name in [(train, 'train'), (dev, 'dev'), (test, 'test')]:

        for domain in map:
            if not os.path.exists('./{}'.format(domain)):
                os.mkdir('./{}'.format(domain))

            if not os.path.exists('./{}/{}'.format(domain, name)):
                os.mkdir('./{}/{}'.format(domain, name))

            with open('./{}/{}/label'.format(domain, name), 'w', encoding='utf-8') as f_label, open('./{}/{}/seq.in'.format(domain, name), 'w', encoding='utf-8') as f_text:

                for data in map[domain]:
                    f_text.write(data[0])
                    f_text.write('\n')
                    f_label.write(data[1])
                    f_label.write('\n')

def process_full_domain_data():
    input_file = './original_data/data_full.json'

    jsn = json.load(open(input_file, 'r'))

    train = []
    dev = []
    test = []
            
    for data_type in jsn:
        if 'oos' in data_type:
            continue
        
        if 'train' in data_type:
            list = train
        elif 'val' in data_type:
            list = dev
        elif 'test' in data_type:
            list = test
        else:
            assert False
            
        for data in jsn[data_type]:
            list.append(data)

    for list, name in [(train, 'train'), (dev, 'dev'), (test, 'test')]:

        if not os.path.exists('./all'):
            os.mkdir('./all')

        if not os.path.exists('./all/{}'.format(name)):
            os.mkdir('./all/{}'.format(name))
            
        with open('./all/{}/label'.format(name), 'w', encoding='utf-8') as f_label, open('./all/{}/seq.in'.format(name), 'w', encoding='utf-8') as f_text:

            for data in list:
                f_text.write(data[0])
                f_text.write('\n')
                f_label.write(data[1])
                f_label.write('\n')

                    
if __name__ == '__main__':
    process_single_domain_data()
    process_full_domain_data()
