# Copyright 2020, Salesforce.com, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

ENTAILMENT = 'entailment'
NON_ENTAILMENT = 'non_entailment'
labelMap = {'1': ENTAILMENT,
            '0': NON_ENTAILMENT,
            'neutral': NON_ENTAILMENT,
            'entailment': ENTAILMENT,
            'contradiction': NON_ENTAILMENT,
            '-': NON_ENTAILMENT}

def main():
    # Training data
    f_out = open('./all_nli.train.txt', 'w')

    for fileName in ['./snli.train.txt', './mnli.train.txt', './wnli.train.txt']:
        with open(fileName, 'r') as f:

            for line in f:
                line = line.rstrip().lower()
                fields = line.split('\t')

                assert len(fields) == 3

                if fields[2] not in labelMap:
                    print(fileName, line)
                    continue

                f_out.write(fields[0]+'\t'+fields[1]+'\t'+labelMap[fields[2]]+'\n')
    f_out.close()

    # Dev data
    f_out = open('./all_nli.dev.txt', 'w')

    for fileName in ['./snli.dev.txt', './mnli.dev.txt', './wnli.dev.txt']:
        with open(fileName, 'r') as f, open('{}_processed.txt'.format(fileName), 'w') as f_sep:

            for line in f:
                line = line.rstrip().lower()
                fields = line.split('\t')

                assert len(fields) == 3

                if fields[2] not in labelMap:
                    print(fileName, line)
                    continue

                f_out.write(fields[0]+'\t'+fields[1]+'\t'+labelMap[fields[2]]+'\n')
                f_sep.write(fields[0]+'\t'+fields[1]+'\t'+labelMap[fields[2]]+'\n')            
    f_out.close()

if __name__ == '__main__':
    main()
