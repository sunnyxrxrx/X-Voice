# SRP Preprocess补充说明 (for Multilingual F5-TTS)
## 数据集大小
- 训练集：从多语言数据集里每个语言抽250h的数据出来，用作计算SRP
- 验证集：从每个语言再抽100条sample出来，做验证集

## filter
### 1. 字符串验证函数
下面这个是Emilia的`字符串验证函数`, 核心作用是确保保留的训练样本中对应的transcript必须只有 **正常的字符**和 **符合要求的标点** `valid_punctuation`

```py
def check_valid_chars(input_str):
    valid_punctuation = '\'",.?!;。，'
    for c in input_str:
        # Check if it's a Latin letter (including uppercase and lowercase)
        if c.isalpha() and ord(c) <= 127:
            continue
        # Check if it's a Chinese character
        if '\u3100' <= c <= '\u9fff':
            continue
        # Check if it's a specified punctuation mark
        if c in valid_punctuation:
            continue
        if c == ' ':
            continue
        return False
        
    return True
```

## syllables计算
1. 先用`pyphen.Pyphen`计算过滤后符合要求sample的syllable数（特殊符号 和。标点符号不包括在音节里）
2. 然后用map_to_class转换成离散类别
   ```py
   def map_to_class(speed, delta=0.25):
       return round(speed / delta) * delta
   ```

## 输出的文件
1. "raw.arrow" 给训练集
2. "raw_val.arrow" 给验证集
3. "duration.json" 给训练集
4. "duration_val.json" 给验证集

arrow的四个字段：
- audio_path：音频绝对路径
- text：文本内容
- duration：时长ground truth
- speed_syllables：计算并映射后的离散类别