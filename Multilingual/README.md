
## 训练集的准备

```
python src/f5_tts/train/datasets/prepare_ipa.py --tokenizer ipa_v3 --dataset_name multilingual_clean
```
- 要把该程序的``inp_dir``改到训练数据metadata所在的目录
  
这会读取metadata文件，生成 ``raw.arrow``, ``duration.json``, ``vocab.txt``
- ``tokenizer``一般选ipa_v3 
  - ipa为第一版，没有加入pinyin
  - ipa_v2在得到ipa序列后像char一样读取，即拆开ai，ei等复音节，效果不太好
  - ipa_v3加入了pinyin，且保留了原始的复音节，仅拆分音节和1，2，3等音调
  - ipa_v4在v3的基础上，对每个单词识别了语种再转换
  - ipa_v5用作在ipa token后面加语言后缀这样的注入形式
- ``dataset_name``需要以 multilingual 开头

## 训练
- 关于yaml文件，在F5TTS_v1_Base的基础上：
  - ``languages``: 将所有语言的简称通过列表形式传进去
  - ``infill_lang_type``: language id的注入方式
    - add_only: 直接和时间相加
    - concat: 和时间拼接
    - token_concat: 逐token拼接
    - ada: 逐token相加调制，没有深入尝试过
    - 如果使用加在ipa token后面加后缀的形式，这里可以选择concat，然后注入通过使用ipa_v5来实现
  - ``use_swiglu`` 和 ``use_rmsnorm``: 基于 linghting dit的改进，设置为true后sim会好一点，但参数量会大，可能要改batch size
  - ``use_ctc``: 是否加上ctc loss，一般选false
  - ``lang_dim``: language id embedding层的维度，默认为与text embedding相同，但为了前向支持，代码做了一些修改，请显式指定
  - ``lang_dropout_prob``: 丢弃language id 的概率，这里丢弃实现在创建language id embedding时增加一个虚拟的语言，drop的时候将其embed到这个虚拟语言

- 将``src/f5_tts/model/dataset.py``的``root_dir``改到metadata所在的目录

- 使用bf16训练

- ckpt参见 https://huggingface.co/XRXRX/Multilingual-F5-TTS ，配置可以参考config中对应名字的文件
  - v4: fp16训练，全局时间concat注入，scale到约500M，rmsnorm，swiglu
  - v5: bf16训练，全局时间concat注入
  - tkcat: bf16训练，逐token concat注入，没有加入drop的逻辑
  - langipa: bf16训练，ipa加后缀的形式注入
  - tkncatv2: bf16训练，在tkncat基础上，调整初始化，并加入随机丢弃language id，概率0.2(待更新)

## 测试集的准备
```
python get_testset_sample10.py
```
- 需要在下好lemas的测试集后，将该程序的``TESTSET_RAW_ROOT``改为测试集所在位置
- 这个程序选取10个样本作为参考，其他作为目标；如果想要全部样本互相作为参考和目标，运行
```
python get_testset.py
```

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
  german-transliterate 
  compound-split
  ```

```
bash src/f5_tts/eval/eval_multilingual.sh
```
需要修改这个脚本前面几行的配置
- ``dataset``: 测试集名称，可选``lemas_eval_new``, ``lemas_eval``，前者选取10个样本为参考，后者全部样本互为参考和目标；预留选项``cv3_eval``，虽然是叫cv3，但实际是想我们自己基于fleurs等造一个支持更多语言的测试集。
- ``ckpt``: checkpoint的步数
- ``exp_name``: 需跟yaml一致
- ``test_set``: 一个字符串，代表要评测的语言，如：``"zh es fr pt en it de vi id"``（lemas的测试机只支持这9种）

此外，还可以修改 ``eval_infer_batch`` 传入的参数
- ``--normalize_text``: 是否进行文本正则化。正则化模块参见``text_normalizer.py``
- ``--usesp -ns "SpeedPredict_Base" -cs 20000``: 是否使用语速预测器
- ``--reverse``: 是否将参考和目标对换位置