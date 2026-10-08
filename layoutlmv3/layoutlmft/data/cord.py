'''
Reference: https://huggingface.co/datasets/pierresi/cord/blob/main/cord.py
'''


import json
import os
from pathlib import Path
import datasets
from layoutlmft.data.image_utils import load_image, normalize_bbox
logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
"""
_DESCRIPTION = """\
https://github.com/clovaai/cord/
"""

def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box

def _get_drive_url(url):
    base_url = 'https://drive.google.com/uc?id='
    split_url = url.split('/')
    return base_url + split_url[5]

_URLS = [
    _get_drive_url("https://drive.google.com/file/d/1MqhTbcj-AHXOqYoeoh12aRUwIprzTJYI/"),
    _get_drive_url("https://drive.google.com/file/d/1wYdp5nC9LnHQZ2FcmOoC0eClyWvcuARU/")
    # If you failed to download the dataset through the automatic downloader,
    # you can download it manually and modify the code to get the local dataset.
    # Or you can use the following links. Please follow the original LICENSE of CORD for usage.
    # "https://layoutlm.blob.core.windows.net/cord/CORD-1k-001.zip",
    # "https://layoutlm.blob.core.windows.net/cord/CORD-1k-002.zip"
]

class CordConfig(datasets.BuilderConfig):
    """BuilderConfig for CORD"""
    def __init__(self, data_dir=None, **kwargs):
        """BuilderConfig for CORD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CordConfig, self).__init__(**kwargs)
        self.data_dir = data_dir

class Cord(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CordConfig(name="cord", version=datasets.Version("1.0.0"), description="CORD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O","B-MENU.NM","B-MENU.NUM","B-MENU.UNITPRICE","B-MENU.CNT","B-MENU.DISCOUNTPRICE","B-MENU.PRICE","B-MENU.ITEMSUBTOTAL","B-MENU.VATYN","B-MENU.ETC","B-MENU.SUB_NM","B-MENU.SUB_UNITPRICE","B-MENU.SUB_CNT","B-MENU.SUB_PRICE","B-MENU.SUB_ETC","B-VOID_MENU.NM","B-VOID_MENU.PRICE","B-SUB_TOTAL.SUBTOTAL_PRICE","B-SUB_TOTAL.DISCOUNT_PRICE","B-SUB_TOTAL.SERVICE_PRICE","B-SUB_TOTAL.OTHERSVC_PRICE","B-SUB_TOTAL.TAX_PRICE","B-SUB_TOTAL.ETC","B-TOTAL.TOTAL_PRICE","B-TOTAL.TOTAL_ETC","B-TOTAL.CASHPRICE","B-TOTAL.CHANGEPRICE","B-TOTAL.CREDITCARDPRICE","B-TOTAL.EMONEYPRICE","B-TOTAL.MENUTYPE_CNT","B-TOTAL.MENUQTY_CNT","I-MENU.NM","I-MENU.NUM","I-MENU.UNITPRICE","I-MENU.CNT","I-MENU.DISCOUNTPRICE","I-MENU.PRICE","I-MENU.ITEMSUBTOTAL","I-MENU.VATYN","I-MENU.ETC","I-MENU.SUB_NM","I-MENU.SUB_UNITPRICE","I-MENU.SUB_CNT","I-MENU.SUB_PRICE","I-MENU.SUB_ETC","I-VOID_MENU.NM","I-VOID_MENU.PRICE","I-SUB_TOTAL.SUBTOTAL_PRICE","I-SUB_TOTAL.DISCOUNT_PRICE","I-SUB_TOTAL.SERVICE_PRICE","I-SUB_TOTAL.OTHERSVC_PRICE","I-SUB_TOTAL.TAX_PRICE","I-SUB_TOTAL.ETC","I-TOTAL.TOTAL_PRICE","I-TOTAL.TOTAL_ETC","I-TOTAL.CASHPRICE","I-TOTAL.CHANGEPRICE","I-TOTAL.CREDITCARDPRICE","I-TOTAL.EMONEYPRICE","I-TOTAL.MENUTYPE_CNT","I-TOTAL.MENUQTY_CNT"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="https://github.com/clovaai/cord/",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir:
            data_root = Path(self.config.data_dir)
            dest = data_root / "CORD" if (data_root / "CORD").is_dir() else data_root
        else:
            downloaded_file = dl_manager.download_and_extract(_URLS)
            first_root = Path(downloaded_file[0])
            second_root = Path(downloaded_file[1])
            dest = first_root / "CORD"
            second_cord = second_root / "CORD"

            if not dest.is_dir():
                raise ValueError(
                    "The first CORD download was not extracted as a directory. "
                    "The source may have returned an HTML/error page instead of a ZIP. "
                    "Download and extract both CORD archives manually, then pass their parent "
                    "directory with --data_dir."
                )
            if not second_cord.is_dir():
                raise ValueError(
                    "The second CORD download was not extracted as a directory. "
                    "The source may have returned an HTML/error page instead of a ZIP. "
                    "Download and extract both CORD archives manually, then pass their parent "
                    "directory with --data_dir."
                )

            # The second archive contains examples that complement the first archive.
            for split in ["train", "dev", "test"]:
                for file_type in ["image", "json"]:
                    if split == "test" and file_type == "json":
                        continue
                    source_dir = second_cord / split / file_type
                    target_dir = dest / split / file_type
                    if not source_dir.is_dir():
                        raise ValueError(f"Expected CORD directory was not found: {source_dir}")
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for source_file in source_dir.iterdir():
                        os.replace(str(source_file), str(target_dir / source_file.name))

        for split in ["train", "dev", "test"]:
            if not (dest / split / "image").is_dir():
                raise ValueError(f"Expected CORD image directory was not found: {dest / split / 'image'}")
            if split != "test" and not (dest / split / "json").is_dir():
                raise ValueError(f"Expected CORD annotation directory was not found: {dest / split / 'json'}")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest/"train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dest/"dev"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest/"test"}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "json")
        img_dir = os.path.join(filepath, "image")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            words = []
            bboxes = []
            ner_tags = []
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["valid_line"]:
                cur_line_bboxes = []
                line_words, label = item["words"], item["category"]
                line_words = [w for w in line_words if w["text"].strip() != ""]
                if len(line_words) == 0:
                    continue
                if label == "other":
                    for w in line_words:
                        words.append(w["text"])
                        ner_tags.append("O")
                        cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
                else:
                    words.append(line_words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(quad_to_box(line_words[0]["quad"]), size))
                    for w in line_words[1:]:
                        words.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        cur_line_bboxes.append(normalize_bbox(quad_to_box(w["quad"]), size))
                # by default: --segment_level_layout 1
                # if do not want to use segment_level_layout, comment the following line
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)
            # yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}
            yield guid, {"id": str(guid), "words": words, "bboxes": bboxes, "ner_tags": ner_tags,
                         "image": image, "image_path": image_path}
