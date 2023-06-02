#数据说明：
##从Textfooler和Clare中获取对抗所用的数据，每个数据集包含1000条数据.
    包括：
    -AG_news
    -Dbpedia
    -fake_news
    -imdb
    -mnli
    -qnli
    -sst-2
    -yahoo
    -yelp
数据集对应的微调训练模型（受害模型），需从[Huggingface TextAttack](https://huggingface.co/textattack) 下载
##训练用数据：
    XXX.csv:逗号分隔
##adv对抗数据：
    XXX.tsv:\t分隔 
    需要三列label text1 text2
    每个类别均等采样

