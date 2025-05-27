import itertools

DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256 = (
    "60be061226930524958b5465c8c04af3d7c03bcb0beb66454f5da9f792e3cf2a"
)

TABLE_DETECTION_RAPIDOCR_MODEL_SHA3_256 = (
    "062f4619afe91b33147c033acadecbb53f2a7b99ac703d157b96d5b10948da5e"
)

TIKTOKEN_CACHES = {
    "fb374d419588a4632f3f557e76b4b70aebbca790": "cb04bcda5782cfbbe77f2f991d92c0ea785d9496ef1137c91dfc3c8c324528d6"
}

FONT_METADATA_URL = {
    "github": "https://raw.githubusercontent.com/funstory-ai/BabelDOC-Assets/refs/heads/main/font_metadata.json",
    "huggingface": "https://huggingface.co/datasets/awwaawwa/BabelDOC-Assets/resolve/main/font_metadata.json?download=true",
    "hf-mirror": "https://hf-mirror.com/datasets/awwaawwa/BabelDOC-Assets/resolve/main/font_metadata.json?download=true",
    "modelscope": "https://www.modelscope.cn/datasets/awwaawwa/BabelDOCAssets/resolve/master/font_metadata.json",
}

FONT_URL_BY_UPSTREAM = {
    "github": lambda name: f"https://raw.githubusercontent.com/funstory-ai/BabelDOC-Assets/refs/heads/main/fonts/{name}",
    "huggingface": lambda name: f"https://huggingface.co/datasets/awwaawwa/BabelDOC-Assets/resolve/main/fonts/{name}?download=true",
    "hf-mirror": lambda name: f"https://hf-mirror.com/datasets/awwaawwa/BabelDOC-Assets/resolve/main/fonts/{name}?download=true",
    "modelscope": lambda name: f"https://www.modelscope.cn/datasets/awwaawwa/BabelDOCAssets/resolve/master/fonts/{name}",
}

DOC_LAYOUT_ONNX_MODEL_URL = {
    "huggingface": "https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx/resolve/main/doclayout_yolo_docstructbench_imgsz1024.onnx?download=true",
    "hf-mirror": "https://hf-mirror.com/wybxc/DocLayout-YOLO-DocStructBench-onnx/resolve/main/doclayout_yolo_docstructbench_imgsz1024.onnx?download=true",
    "modelscope": "https://www.modelscope.cn/models/AI-ModelScope/DocLayout-YOLO-DocStructBench-onnx/resolve/master/doclayout_yolo_docstructbench_imgsz1024.onnx",
}

TABLE_DETECTION_RAPIDOCR_MODEL_URL = {
    "huggingface": "https://huggingface.co/spaces/RapidAI/RapidOCR/resolve/main/models/text_det/ch_PP-OCRv4_det_infer.onnx",
    "hf-mirror": "https://hf-mirror.com/spaces/RapidAI/RapidOCR/resolve/main/models/text_det/ch_PP-OCRv4_det_infer.onnx",
    "modelscope": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_infer.onnx",
}

# from https://github.com/funstory-ai/BabelDOC-Assets/blob/main/font_metadata.json
EMBEDDING_FONT_METADATA = {
    "GoNotoKurrent-Bold.ttf": {
        "ascent": 1069,
        "bold": 1,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "GoNotoKurrent-Bold.ttf",
        "font_name": "Go Noto Kurrent-Bold Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "000b37f592477945b27b7702dcad39f73e23e140e66ddff9847eb34f32389566",
        "size": 15303772,
    },
    "GoNotoKurrent-Regular.ttf": {
        "ascent": 1069,
        "bold": 0,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "GoNotoKurrent-Regular.ttf",
        "font_name": "Go Noto Kurrent-Regular Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "4324a60d507c691e6efc97420647f4d2c2d86d9de35009d1c769861b76074ae6",
        "size": 15515760,
    },
    "KleeOne-Regular.ttf": {
        "ascent": 1160,
        "bold": 0,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "KleeOne-Regular.ttf",
        "font_name": "Klee One Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "8585c29f89b322d937f83739f61ede5d84297873e1465cad9a120a208ac55ce0",
        "size": 8724704,
    },
    "LXGWWenKaiGB-Regular.ttf": {
        "ascent": 928,
        "bold": 0,
        "descent": -256,
        "encoding_length": 2,
        "file_name": "LXGWWenKaiGB-Regular.ttf",
        "font_name": "LXGW WenKai GB Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "b563a5e8d9db4cd15602a3a3700b01925e80a21f99fb88e1b763b1fb8685f8ee",
        "size": 19558756,
    },
    "LXGWWenKaiMonoTC-Regular.ttf": {
        "ascent": 928,
        "bold": 0,
        "descent": -241,
        "encoding_length": 2,
        "file_name": "LXGWWenKaiMonoTC-Regular.ttf",
        "font_name": "LXGW WenKai Mono TC Regular",
        "italic": 0,
        "monospace": 1,
        "serif": 1,
        "sha3_256": "596b278d11418d374a1cfa3a50cbfb82b31db82d3650cfacae8f94311b27fdc5",
        "size": 13115416,
    },
    "LXGWWenKaiTC-Regular.ttf": {
        "ascent": 928,
        "bold": 0,
        "descent": -256,
        "encoding_length": 2,
        "file_name": "LXGWWenKaiTC-Regular.ttf",
        "font_name": "LXGW WenKai TC Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "66ccd0ffe8e56cd585dabde8d1292c3f551b390d8ed85f81d7a844825f9c2379",
        "size": 13100328,
    },
    "MaruBuri-Regular.ttf": {
        "ascent": 800,
        "bold": 0,
        "descent": -200,
        "encoding_length": 2,
        "file_name": "MaruBuri-Regular.ttf",
        "font_name": "MaruBuri Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "abb672dde7b89e06914ce27c59159b7a2933f26207bfcc47981c67c11c41e6d1",
        "size": 3268988,
    },
    "NotoSans-Bold.ttf": {
        "ascent": 1069,
        "bold": 1,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "NotoSans-Bold.ttf",
        "font_name": "Noto Sans Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "ecd38d472c1cad07d8a5dffd2b5a0f72edcd40fff2b4e68d770da8f2ef343a82",
        "size": 630964,
    },
    "NotoSans-BoldItalic.ttf": {
        "ascent": 1069,
        "bold": 1,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "NotoSans-BoldItalic.ttf",
        "font_name": "Noto Sans Bold Italic",
        "italic": 1,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "0b6c690a4a6b7d605b2ecbde00c7ac1a23e60feb17fa30d8b972d61ec3ff732b",
        "size": 644340,
    },
    "NotoSans-Italic.ttf": {
        "ascent": 1069,
        "bold": 0,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "NotoSans-Italic.ttf",
        "font_name": "Noto Sans Italic",
        "italic": 1,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "830652f61724c017e5a29a96225b484a2ccbd25f69a1b3f47e5f466a2dbed1ad",
        "size": 642344,
    },
    "NotoSans-Regular.ttf": {
        "ascent": 1069,
        "bold": 0,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "NotoSans-Regular.ttf",
        "font_name": "Noto Sans Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "7dfe2bbf97dc04c852d1223b220b63430e6ad03b0dbb28ebe6328a20a2d45eb8",
        "size": 629024,
    },
    "NotoSerif-Bold.ttf": {
        "ascent": 1069,
        "bold": 1,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "NotoSerif-Bold.ttf",
        "font_name": "Noto Serif Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "28d88d924285eadb9f9ce49f2d2b95473f89a307b226c5f6ebed87a654898312",
        "size": 506864,
    },
    "NotoSerif-BoldItalic.ttf": {
        "ascent": 1069,
        "bold": 1,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "NotoSerif-BoldItalic.ttf",
        "font_name": "Noto Serif Bold Italic",
        "italic": 1,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "b69ee56af6351b2fb4fbce623f8e1c1f9fb19170686a9e5db2cf260b8cf24ac7",
        "size": 535724,
    },
    "NotoSerif-Italic.ttf": {
        "ascent": 1069,
        "bold": 0,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "NotoSerif-Italic.ttf",
        "font_name": "Noto Serif Italic",
        "italic": 1,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "9b7773c24ab8a29e3c1c03efa4ab652d051e4c209134431953463aa946d62868",
        "size": 535340,
    },
    "NotoSerif-Regular.ttf": {
        "ascent": 1069,
        "bold": 0,
        "descent": -293,
        "encoding_length": 2,
        "file_name": "NotoSerif-Regular.ttf",
        "font_name": "Noto Serif Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "c2bbe984e65bafd3bcd38b3cb1e1344f3b7b79d6beffc7a3d883b57f8358559d",
        "size": 504932,
    },
    "SourceHanSansCN-Bold.ttf": {
        "ascent": 1160,
        "bold": 1,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansCN-Bold.ttf",
        "font_name": "Source Han Sans CN Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "82314c11016a04ef03e7afd00abe0ccc8df54b922dee79abf6424f3002a31825",
        "size": 10174460,
    },
    "SourceHanSansCN-Regular.ttf": {
        "ascent": 1160,
        "bold": 0,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansCN-Regular.ttf",
        "font_name": "Source Han Sans CN Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "b45a80cf3650bfc62aa014e58243c6325e182c4b0c5819e41a583c699cce9a8f",
        "size": 10397552,
    },
    "SourceHanSansHK-Bold.ttf": {
        "ascent": 1160,
        "bold": 1,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansHK-Bold.ttf",
        "font_name": "Source Han Sans HK Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "3eecd57457ba9a0fbad6c794f40e7ae704c4f825091aef2ac18902ffdde50608",
        "size": 6856692,
    },
    "SourceHanSansHK-Regular.ttf": {
        "ascent": 1160,
        "bold": 0,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansHK-Regular.ttf",
        "font_name": "Source Han Sans HK Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "5fe4141f9164c03616323400b2936ee4c8265314492e2b822c3a6fbfb63ffe08",
        "size": 6999792,
    },
    "SourceHanSansJP-Bold.ttf": {
        "ascent": 1160,
        "bold": 1,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansJP-Bold.ttf",
        "font_name": "Source Han Sans JP Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "fb05bd84d62e8064117ee357ab6a4481e1cde931e8e984c0553c8c4b09dc3938",
        "size": 5603068,
    },
    "SourceHanSansJP-Regular.ttf": {
        "ascent": 1160,
        "bold": 0,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansJP-Regular.ttf",
        "font_name": "Source Han Sans JP Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "722cfbdcc0fd83fe07a3d1b10e9e64343c924a351d02cfe8dbb6ec4c6bc38230",
        "size": 5723960,
    },
    "SourceHanSansKR-Bold.ttf": {
        "ascent": 1160,
        "bold": 1,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansKR-Bold.ttf",
        "font_name": "Source Han Sans KR Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "02959eb2c1eea0786a736aeb50b6e61f2ab873cd69c659389b7511f80f734838",
        "size": 5858892,
    },
    "SourceHanSansKR-Regular.ttf": {
        "ascent": 1160,
        "bold": 0,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansKR-Regular.ttf",
        "font_name": "Source Han Sans KR Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "aba70109eff718e8f796f0185f8dca38026c1661b43c195883c84577e501adf2",
        "size": 5961704,
    },
    "SourceHanSansTW-Bold.ttf": {
        "ascent": 1160,
        "bold": 1,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansTW-Bold.ttf",
        "font_name": "Source Han Sans TW Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "4a92730e644a1348e87bba7c77e9b462f257f381bd6abbeac5860d8f8306aee6",
        "size": 6883224,
    },
    "SourceHanSansTW-Regular.ttf": {
        "ascent": 1160,
        "bold": 0,
        "descent": -288,
        "encoding_length": 2,
        "file_name": "SourceHanSansTW-Regular.ttf",
        "font_name": "Source Han Sans TW Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "6129b68ff4b0814624cac7edca61fbacf8f4d79db6f4c3cfc46b1c48ea2f81ac",
        "size": 7024812,
    },
    "SourceHanSerifCN-Bold.ttf": {
        "ascent": 1150,
        "bold": 1,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifCN-Bold.ttf",
        "font_name": "Source Han Serif CN Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "77816a54957616e140e25a36a41fc061ddb505a1107de4e6a65f561e5dcf8310",
        "size": 14134156,
    },
    "SourceHanSerifCN-Regular.ttf": {
        "ascent": 1150,
        "bold": 0,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifCN-Regular.ttf",
        "font_name": "Source Han Serif CN Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "c8bf74da2c3b7457c9d887465b42fb6f80d3d84f361cfe5b0673a317fb1f85ad",
        "size": 14047768,
    },
    "SourceHanSerifHK-Bold.ttf": {
        "ascent": 1150,
        "bold": 1,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifHK-Bold.ttf",
        "font_name": "Source Han Serif HK Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "0f81296f22846b622a26f7342433d6c5038af708a32fc4b892420c150227f4bb",
        "size": 9532580,
    },
    "SourceHanSerifHK-Regular.ttf": {
        "ascent": 1150,
        "bold": 0,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifHK-Regular.ttf",
        "font_name": "Source Han Serif HK Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "d5232ec3adf4fb8604bb4779091169ec9bd9d574b513e4a75752e614193afebe",
        "size": 9467292,
    },
    "SourceHanSerifJP-Bold.ttf": {
        "ascent": 1150,
        "bold": 1,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifJP-Bold.ttf",
        "font_name": "Source Han Serif JP Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "a4a8c22e8ec7bb6e66b9caaff1e12c7a52b5a4201eec3d074b35957c0126faef",
        "size": 7811832,
    },
    "SourceHanSerifJP-Regular.ttf": {
        "ascent": 1150,
        "bold": 0,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifJP-Regular.ttf",
        "font_name": "Source Han Serif JP Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "3d1f9933c7f3abc8c285e317119a533e6dcfe6027d1f5f066ba71b3eb9161e9c",
        "size": 7748816,
    },
    "SourceHanSerifKR-Bold.ttf": {
        "ascent": 1150,
        "bold": 1,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifKR-Bold.ttf",
        "font_name": "Source Han Serif KR Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "b071b1aecb042aa779e1198767048438dc756d0da8f90660408abb421393f5cb",
        "size": 12387920,
    },
    "SourceHanSerifKR-Regular.ttf": {
        "ascent": 1150,
        "bold": 0,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifKR-Regular.ttf",
        "font_name": "Source Han Serif KR Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "a85913439f0a49024ca77c02dfede4318e503ee6b2b7d8fef01eb42435f27b61",
        "size": 12459924,
    },
    "SourceHanSerifTW-Bold.ttf": {
        "ascent": 1150,
        "bold": 1,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifTW-Bold.ttf",
        "font_name": "Source Han Serif TW Bold",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "562eea88895ab79ffefab7eabb4d322352a7b1963764c524c6d5242ca456bb6e",
        "size": 9551724,
    },
    "SourceHanSerifTW-Regular.ttf": {
        "ascent": 1150,
        "bold": 0,
        "descent": -286,
        "encoding_length": 2,
        "file_name": "SourceHanSerifTW-Regular.ttf",
        "font_name": "Source Han Serif TW Regular",
        "italic": 0,
        "monospace": 0,
        "serif": 1,
        "sha3_256": "85c1d6460b2e169b3d53ac60f6fb7a219fb99923027d78fb64b679475e2ddae4",
        "size": 9486772,
    },
}

FONT_NAMES = {v["font_name"] for v in EMBEDDING_FONT_METADATA.values()}

CN_FONT_FAMILY = {
    # 手写体
    "script": [
        "LXGWWenKaiGB-Regular.ttf",
    ],
    # 正文字体
    "normal": [
        "SourceHanSerifCN-Bold.ttf",
        "SourceHanSerifCN-Regular.ttf",
        "SourceHanSansCN-Bold.ttf",
        "SourceHanSansCN-Regular.ttf",
    ],
    # 备用字体
    "fallback": [
        "GoNotoKurrent-Regular.ttf",
        "GoNotoKurrent-Bold.ttf",
    ],
    "base": ["SourceHanSansCN-Regular.ttf"],
}

HK_FONT_FAMILY = {
    "script": ["LXGWWenKaiTC-Regular.ttf"],
    "normal": [
        "SourceHanSerifHK-Bold.ttf",
        "SourceHanSerifHK-Regular.ttf",
        "SourceHanSansHK-Bold.ttf",
        "SourceHanSansHK-Regular.ttf",
    ],
    "fallback": [
        "GoNotoKurrent-Regular.ttf",
        "GoNotoKurrent-Bold.ttf",
    ],
    "base": ["SourceHanSansCN-Regular.ttf"],
}

TW_FONT_FAMILY = {
    "script": ["LXGWWenKaiTC-Regular.ttf"],
    "normal": [
        "SourceHanSerifTW-Bold.ttf",
        "SourceHanSerifTW-Regular.ttf",
        "SourceHanSansTW-Bold.ttf",
        "SourceHanSansTW-Regular.ttf",
    ],
    "fallback": [
        "GoNotoKurrent-Regular.ttf",
        "GoNotoKurrent-Bold.ttf",
    ],
    "base": ["SourceHanSansCN-Regular.ttf"],
}

KR_FONT_FAMILY = {
    "script": ["MaruBuri-Regular.ttf"],
    "normal": [
        "SourceHanSerifKR-Bold.ttf",
        "SourceHanSerifKR-Regular.ttf",
        "SourceHanSansKR-Bold.ttf",
        "SourceHanSansKR-Regular.ttf",
    ],
    "fallback": [
        "GoNotoKurrent-Regular.ttf",
        "GoNotoKurrent-Bold.ttf",
    ],
    "base": ["SourceHanSansCN-Regular.ttf"],
}

JP_FONT_FAMILY = {
    "script": ["KleeOne-Regular.ttf"],
    "normal": [
        "SourceHanSerifJP-Bold.ttf",
        "SourceHanSerifJP-Regular.ttf",
        "SourceHanSansJP-Bold.ttf",
        "SourceHanSansJP-Regular.ttf",
    ],
    "fallback": [
        "GoNotoKurrent-Regular.ttf",
        "GoNotoKurrent-Bold.ttf",
    ],
    "base": ["SourceHanSansCN-Regular.ttf"],
}

EN_FONT_FAMILY = {
    "script": [
        "NotoSans-Italic.ttf",
        "NotoSans-BoldItalic.ttf",
        "NotoSerif-Italic.ttf",
        "NotoSerif-BoldItalic.ttf",
    ],
    "normal": [
        "NotoSerif-Regular.ttf",
        "NotoSerif-Bold.ttf",
        "NotoSans-Regular.ttf",
        "NotoSans-Bold.ttf",
    ],
    "fallback": [
        "GoNotoKurrent-Regular.ttf",
        "GoNotoKurrent-Bold.ttf",
    ],
    "base": [
        "NotoSans-Regular.ttf",
    ],
}

ALL_FONT_FAMILY = {
    "CN": CN_FONT_FAMILY,
    "TW": TW_FONT_FAMILY,
    "HK": HK_FONT_FAMILY,
    "KR": KR_FONT_FAMILY,
    "JP": JP_FONT_FAMILY,
    "EN": EN_FONT_FAMILY,
}


def __add_fallback_to_font_family():
    for lang1, family1 in ALL_FONT_FAMILY.items():
        added_font = set()
        for font in itertools.chain.from_iterable(family1.values()):
            added_font.add(font)

        for lang2, family2 in ALL_FONT_FAMILY.items():
            if lang1 != lang2:
                for type_ in family1:
                    for font in family2[type_]:
                        if font not in added_font:
                            family1[type_].append(font)
                            added_font.add(font)


__add_fallback_to_font_family()


def get_font_family(lang_code: str):
    lang_code = lang_code.upper()
    if "KR" in lang_code:
        font_family = KR_FONT_FAMILY
    elif "KO" in lang_code:
        font_family = KR_FONT_FAMILY
    elif "JP" in lang_code:
        font_family = JP_FONT_FAMILY
    elif "HK" in lang_code:
        font_family = HK_FONT_FAMILY
    elif "TW" in lang_code:
        font_family = TW_FONT_FAMILY
    elif "EN" in lang_code:
        font_family = EN_FONT_FAMILY
    elif "CN" in lang_code:
        font_family = CN_FONT_FAMILY
    else:
        font_family = EN_FONT_FAMILY
    verify_font_family(font_family)
    return font_family


def verify_font_family(font_family: str | dict):
    if isinstance(font_family, str):
        font_family = ALL_FONT_FAMILY[font_family]
    for k in font_family:
        if k not in ["script", "normal", "fallback", "base"]:
            raise ValueError(f"Invalid font family: {font_family}")
        for font_file_name in font_family[k]:
            if font_file_name not in EMBEDDING_FONT_METADATA:
                raise ValueError(f"Invalid font file: {font_file_name}")


if __name__ == "__main__":
    for k in ALL_FONT_FAMILY:
        verify_font_family(k)
