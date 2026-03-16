
## 训练集的准备
安装以下依赖，其中后两者可选，只需在ipa_v4中使用
```
phonemizer
pythainlp
pyopenjtalk
g2pk
regex

lingua
spellchecker
```
并安装espeak
```
bash prepare_ipa.sh
```
转ipa
```
python src/f5_tts/train/datasets/prepare_ipa.py --tokenizer ipa_v3 --dataset_name multilingual_clean
```
- 要把该程序的``inp_dir``改到训练数据metadata所在的目录
  
这会读取metadata文件，生成 ``raw.arrow``, ``duration.json``, ``vocab.txt``, ``vocab_stat.txt``
- ``tokenizer``一般选ipa_v6
  - ipa为第一版，没有加入pinyin
  - ipa_v2在得到ipa序列后像char一样读取，即拆开ai，ei等复音节，效果不太好
  - ipa_v3加入了pinyin，且保留了原始的复音节，仅拆分音节和1，2，3等音调
  - ipa_v4在v3的基础上，对每个单词识别了语种再转换
  - ipa_v5用作在ipa token后面加语言后缀这样的注入形式
  - ipa_v6在v3的基础上，增加了日语和韩语的g2p。在跑``prepare_ipa.py``的同时，得出每个ipa出现的频率，对于出现频率极少的ipa，如``nɛ``等，不将其放入词表，在查词表转为idx时，不直接将其映射为``<unk>``，而是将其拆分为``n``和``ɛ``分别映射。
- ``dataset_name``需要以 multilingual 开头

## 训练
- 关于yaml文件，在F5TTS_v1_Base的基础上：
  - ``languages``: 将所有语言的简称通过列表形式传进去
  - ``time_infill_lang_type``: language id在时间的注入方式
    - add_only: 直接和时间相加
    - time_concat: 和时间拼接
  - ``time_infill_lang_type``: language id在text token的注入方式
    - token_concat: 逐token拼接
    - ada: 逐token相加调制
  - ``use_swiglu`` 和 ``use_rmsnorm``: 基于 linghting dit的改进，设置为true后sim会好一点，但参数量会大，可能要改batch size
  - ``use_ctc``: 是否加上ctc loss，一般选False
  - ``lang_dim``, ``lang_dim_in_t``: language id embedding层的维度
  - ``lang_drop_prob``, ``cond_drop_prob``: 只丢弃语言和同时丢弃语言和文本的概率
  - ``share_lang_embed``: 时间和文本是否共用language embedding
  - ``pretrained_path``, ``freeze_update``加载原版ckpt做transfer learning，并可以指定前面几万步冻结DiT参数
- 将``src/f5_tts/model/dataset.py``的``root_dir``改到metadata所在的目录


- 使用bf16训练

- ckpt参见 https://huggingface.co/XRXRX/Multilingual-F5-TTS ，配置可以参考config中对应名字的文件
  - catada: 时间上使用time_concat，文本上使用ada，具体见配置文件
  - catada_with_stress: 在catada的基础上对ipa增加了重音处理，这个可以解决希腊语发音的问题。

## 测试集的准备
```
python get_testset_sample10.py
```
- 需要在下好lemas的测试集后，将该程序的``TESTSET_RAW_ROOT``改为测试集所在位置
- 这个程序选取lemas的10个样本作为参考，其他作为目标；如果想要全部样本互相作为参考和目标，运行
```
python get_mixed_testset.py
```
这个会对文本进行一些规则性的筛选，并对音频文件做VAD和DNSMOS筛选，适合自己构建测试集。
## 推理和评测
- 推理时需要补充以下依赖，在运行``eval_multilingual.sh``时会自动下载
  ```
  pytest-runner
  openai-whisper
  onnxruntime-gpu
  addict
  simplejson
  modelscope
  num2words
  nemo_text_processing
  WeTextProcessing 
  xphonebr 
  ctranslate2==4.5.0
  pyphen
  bg-text-normalizer
  git+https://github.com/TartuNLP/tts_preprocess_et.git
  ```


### intra-lingual
```
bash src/f5_tts/eval/eval_multilingual.sh
```
需要修改这个脚本前面几行的配置
- ``dataset``: 测试集名称，可选``lemas_eval_new``, ``lemas_eval``，前者选取10个样本为参考，后者全部样本互为参考和目标；预留选项``mixed_eval``，我们自己构造的支持更多语言的测试集。
- ``ckpt``: checkpoint的步数
- ``exp_name``: 需跟yaml一致
- ``test_set``: 一个字符串，代表要评测的语言，如：``"zh es fr pt en it de vi id"``（lemas的测试集只支持这9种）



### cross-lingual
```
bash src/f5_tts/eval/eval_cross_lingual.sh
```
配置的修改同intra-lingual的代码，此外，增加``ref_set``的配置
- ``ref_set``: 一个字符串，代表prompt的语言，必须与``test_set``等长。
  - 例如``test_set="en fr", ref_set="fr zh"``，那么将以法语为参考生成英语和以中文为参考生成法语。

### 推理的配置
在上述两个脚本中，可以修改 ``eval_infer_batch`` 传入的参数
- ``--normalize_text``: 是否进行文本正则化。正则化模块参见``text_normalizer.py``
- ``--sp_type / -sp``: 如何预测生成音频的时长。可选``"utf", "syllable", "pretrained"``
  - ``"utf"``: 默认选项。即根据utf-8字节数的比例来预测。
  - ``"syllable"``: 根据文本音节数比例来预测。
  - ``"pretrained"``: 使用预训练的语速预测器。如果选择此选项，需补充实验名``--expnamesp / -ns``和ckpt步数``--ckptstepsp/ -cs``。例如``-ns "SpeedPredict_Base" -cs 20000``
- ``--reverse``: 是否将参考和目标对换位置
- ``--layered``: 是否使用分层cfg推理
- ``--cfg_schedule=linear``, ``--cfg_decay_time=0.6``在推理后期降低cfg强度，精细打磨声学细节