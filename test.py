

import jieba
import codecs

fread = codecs.open("corpus.txt", encoding="GBK")
fwrite = open("result.txt", "a")

lines = fread.readlines()
for line in lines:
    line.replace('\t', '').replace('\n', '').replace(' ', '')
    seg_list = jieba.cut(line, cut_all=False)
    fwrite.write(" ".join(seg_list))

fread.close()
fwrite.close()


