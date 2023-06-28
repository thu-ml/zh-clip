# ZH-CLIP: A Chinese CLIP Model
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/thu-ml/ZH-CLIP)

## Models
You can download **ZH-CLIP** model from [🤗 thu-ml/zh-clip-vit-roberta-large-patch14](https://huggingface.co/thu-ml/zh-clip-vit-roberta-large-patch14). The model structure is shown below:
* Vision encoder network structure is the same as [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14), and initialize with [laion/CLIP-ViT-L-14-laion2B-s32B-b82K](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K).
* Text encoder network struceure is the same as [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) and initialized.
## Results
**Note: We tried to maintain consistency with the majority of comparative models in terms of scale. However, some models used smaller text encoders (eg. CNCLIP), so the following comparisons may not be entirely fair.** 
#### COCO-CN Retrieval (Official Test Set):
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4">Text-to-Image</th>
      <th colspan="4">Image-to-Text</th>
    </tr>
    <tr>
      <th>R@1</th>
      <th>R@5</th>
      <th>R@10</th>
      <th>Mean</th>
      <th>R@1</th>
      <th>R@5</th>
      <th>R@10</th>
      <th>Mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Clip-Chinese</td>
      <td>22.60</td>
      <td>50.04</td>
      <td>65.24</td>
      <td>45.96</td>
      <td>22.8</td>
      <td>49.8</td>
      <td>64.1</td>
      <td>45.57</td>
    </tr>
    <tr>
      <td>mclip</td>
      <td>56.51</td>
      <td>83.57</td>
      <td>90.79</td>
      <td>76.95</td>
      <td>59.9</td>
      <td>87.3</td>
      <td>94.1</td>
      <td>80.43</td>
    </tr>
    <tr>
      <td>Taiyi-CLIP</td>
      <td>52.52</td>
      <td>81.10</td>
      <td>89.93</td>
      <td>74.52</td>
      <td>45.80</td>
      <td>75.80</td>
      <td>88.10</td>
      <td>69.90</td>
    </tr>
    <tr>
      <td>CN-CLIP</td>
      <td>64.10</td>
      <td>88.79</td>
      <td>94.40</td>
      <td>82.43</td>
      <td>61.00</td>
      <td>84.40</td>
      <td>93.10</td>
      <td>79.5</td>
    </tr>
    <tr>
      <td>altclip-xlmr-l</td>
      <td>62.87</td>
      <td>87.18</td>
      <td>94.01</td>
      <td>81.35</td>
      <td>63.3</td>
      <td>88.3</td>
      <td>95.3</td>
      <td>82.3</td>
    </tr>
    <tr>
      <td>ZH-CLIP</td>
      <td><strong>68.00</strong></td>
      <td><strong>89.46</strong></td>
      <td><strong>95.44</strong></td>
      <td><strong>84.30</strong></td>
      <td><strong>68.50</strong></td>
      <td><strong>90.10</strong></td>
      <td><strong>96.50</strong></td>
      <td><strong>85.03</strong></td>
    </tr>
  </tbody>
</table>

#### Flickr30K-CN Retrieval (Official Test Set):
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4">Text-to-Image</th>
      <th colspan="4">Image-to-Text</th>
    </tr>
    <tr>
      <th>R@1</th>
      <th>R@5</th>
      <th>R@10</th>
      <th>Mean</th>
      <th>R@1</th>
      <th>R@5</th>
      <th>R@10</th>
      <th>Mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Clip-Chinese</td>
      <td>17.76</td>
      <td>40.34</td>
      <td>51.88</td>
      <td>36.66</td>
      <td>30.4</td>
      <td>55.30</td>
      <td>67.10</td>
      <td>50.93</td>
    </tr>
    <tr>
      <td>mclip</td>
      <td>62.3</td>
      <td>86.42</td>
      <td>92.58</td>
      <td>80.43</td>
      <td>84.4</td>
      <td>97.3</td>
      <td>98.9</td>
      <td>93.53</td>
    </tr>
    <tr>
      <td>Taiyi-CLIP</td>
      <td>53.5</td>
      <td>80.5</td>
      <td>87.24</td>
      <td>73.75</td>
      <td>65.4</td>
      <td>90.6</td>
      <td>95.7</td>
      <td>83.9</td>
    </tr>
    <tr>
      <td>CN-CLIP</td>
      <td>67.98</td>
      <td>89.54</td>
      <td>94.46</td>
      <td>83.99</td>
      <td>81.2</td>
      <td>96.6</td>
      <td>98.2</td>
      <td>92.0</td>
    </tr>
    <tr>
      <td>altclip-xlmr-l</td>
      <td>69.16</td>
      <td>89.94</td>
      <td><strong>94.5</strong></td>
      <td>84.53</td>
      <td>85.1</td>
      <td><strong>97.7</strong></td>
      <td><strong>99.2</strong></td>
      <td>94.0</td>
    </tr>
    <tr>
      <td>ZH-CLIP</td>
      <td><strong>69.64</strong></td>
      <td><strong>90.14</strong></td>
      <td>94.3</td>
      <td><strong>84.69</strong></td>
      <td><strong>86.6</strong></td>
      <td>97.6</td>
      <td>98.8</td>
      <td><strong>94.33</strong></td>
    </tr>
  </tbody>
</table>


#### Muge Text-to-Image Retrieval (Official Validation Set):
<table>
  <thead>
    <tr>
        <th rowspan="2">Model</th>
        <th colspan="4">Text-to-Image</th>
    </tr>
    <tr>
        <th>R@1</th>
        <th>R@5</th>
        <th>R@10</th>
        <th>Mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>Clip-Chinese</td>
        <td>15.06</td>
        <td>34.96</td>
        <td>46.21</td>
        <td>32.08</td>
    </tr>
    <tr>
        <td>mclip</td>
        <td>22.34</td>
        <td>41.15</td>
        <td>50.26</td>
        <td>37.92</td>
    </tr>
    <tr>
        <td>Taiyi-CLIP</td>
        <td>42.09</td>
        <td>67.75</td>
        <td>77.21</td>
        <td>62.35</td>
    </tr>
    <tr>
        <td>cn-clip</td>
        <td>56.25</td>
        <td><strong>79.87</strong></td>
        <td>86.50</td>
        <td>74.21</td>
    </tr>
    <tr>
        <td>altclip-xlmr-l</td>
        <td>29.69</td>
        <td>49.92</td>
        <td>58.87</td>
        <td>46.16</td>
    </tr>
    <tr>
        <td>ZH-CLIP</td>
        <td><strong>56.75</strong></td>
        <td>79.75</td>
        <td><strong>86.66</strong></td>
        <td><strong>74.38</strong></td>
    </tr>
  </tbody>
</table>

#### Zero-shot Image Classification:
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="11">Zero-shot Classification (ACC1)</th>
    </tr>
    <tr>
      <th>CIFAR10</th>
      <th>CIFAR100</th>
      <th>DTD</th>
      <th>EuroSAT</th>
      <th>FER</th>
      <th>FGVC</th>
      <th>KITTI</th>
      <th>MNIST</th>
      <th>PC</th>
      <th>VOC</th>
      <th>ImageNet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Clip-Chinese</td>
      <td>86.85</td>
      <td>44.21</td>
      <td>18.40</td>
      <td>34.86</td>
      <td>14.21</td>
      <td>3.87</td>
      <td>32.63</td>
      <td>14.37</td>
      <td>52.49</td>
      <td>67.73</td>
      <td>22.22</td>
    </tr>
    <tr>
      <td>mclip</td>
      <td>92.88</td>
      <td>65.54</td>
      <td>29.57</td>
      <td>46.76</td>
      <td>41.18</td>
      <td>7.20</td>
      <td>23.21</td>
      <td>52.80</td>
      <td>51.64</td>
      <td>77.56</td>
      <td>42.99</td>
    </tr>
    <tr>
      <td>Taiyi-CLIP</td>
      <td>95.62</td>
      <td>73.30</td>
      <td>40.69</td>
      <td><strong>61.62</strong></td>
      <td>36.22</td>
      <td>13.98</td>
      <td><strong>41.21</strong></td>
      <td><strong>73.91</strong></td>
      <td>50.02</td>
      <td>75.28</td>
      <td>49.82</td>
    </tr>
    <tr>
      <td>CN-CLIP</td>
      <td>94.75</td>
      <td>75.04</td>
      <td>44.73</td>
      <td>52.34</td>
      <td>48.57</td>
      <td>20.55</td>
      <td>20.11</td>
      <td>61.99</td>
      <td><strong>62.59</strong></td>
      <td><strong>79.12</strong></td>
      <td>53.40</td>
    </tr>
    <tr>
      <td>Altclip-xlmr-l</td>
      <td>95.49</td>
      <td>77.29</td>
      <td>42.07</td>
      <td>56.96</td>
      <td><strong>51.52</strong></td>
      <td><strong>26.85</strong></td>
      <td>24.89</td>
      <td>65.68</td>
      <td>50.02</td>
      <td>77.99</td>
      <td><strong>59.21</strong></td>
    </tr>
    <tr>
      <td>ZH-CLIP</td>
      <td><strong>97.08</strong></td>
      <td><strong>80.73</strong></td>
      <td><strong>47.66</strong></td>
      <td>51.58</td>
      <td>48.48</td>
      <td>20.73</td>
      <td>20.11</td>
      <td>61.94</td>
      <td>62.31</td>
      <td>78.07</td>
      <td>56.87</td>
    </tr>
  </tbody>
</table>

## Getting Started
### Dependency
* python >= 3.9
* pip install -r requirements.txt
### Inference
```python
from PIL import Image
import requests
from models.zhclip import ZhCLIPProcessor, ZhCLIPModel

version = 'thu-ml/zh-clip-vit-roberta-large-patch14'
model = ZhCLIPModel.from_pretrained(version)
processor = ZhCLIPProcessor.from_pretrained(version)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=["一只猫", "一只狗"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
image_features = outputs.image_features
text_features = outputs.text_features
text_probs = (image_features @ text_features.T).softmax(dim=-1)
```
### Other Chinese CLIP Models
In addition, to compare the effectiveness of different methods, the inference methods of other Chinese CLIP models have been integrated. For the convenience of use, the inference code has also been made public, and please contact us if there is any infringement. The code only implements models at the same level as clip-vit-large-patch14, but it may be adapted for the use of more different versions of models in the future.
| # | model | alias |
| :----: | :---------- | :---------- |
| 0 | [ZH-CLIP](https://github.com/thu-ml/zh-clip) | zhclip |
| 1	| [AltCLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP) | altclip |
| 2	| [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)	| cnclip |
| 3	| [TaiyiCLIP](https://github.com/IDEA-CCNL/Fengshenbang-LM)	| taiyiclip |
| 4	| [Multilingual-CLIP](https://github.com/FreddeFrallan/Multilingual-CLIP)	| mclip |
| 5	| [CLIP-Chinese](https://github.com/yangjianxin1/CLIP-Chinese)	| clip-chinese |

Usage in [inference.py](https://github.com/thu-ml/zh-clip/blob/main/inference.py)


