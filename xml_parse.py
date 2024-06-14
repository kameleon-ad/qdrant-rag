import re
from io import BytesIO
from pathlib import Path
import pandas as pd
from lxml import etree


class NiceClass:
    def __init__(self):
        self.classNumber = 0
        self.isGoodOrService = ""
        self.id = ""
        self.classTexts = None
        self.goodOrServices = []

    def toDict(self):
        result = {}
        result["class_id"] = int(self.classNumber)
        result["heading"] = self.classTexts.heading
        result["introduction"] = self.classTexts.explanatoryNote.introduction
        result["include"] = self.classTexts.explanatoryNote.includesParticular
        result["exclude"] = self.classTexts.explanatoryNote.excludesPerticular
        goodslist = []
        for good in self.goodOrServices:
            goodslist.extend(good.labels)
        result["good_or_service"] = goodslist
        return result


class GoodOrService:
    def __init__(self):
        self.basicNumber = ""
        self.id = ""
        self.labels = []


class ClassText:
    def __init__(self):
        self.id = ""
        self.heading = []
        self.explanatoryNote = ExplanatoryNote()

    def addHeadingItem(self, headingItem):
        self.heading.append(headingItem)


class ExplanatoryNote:
    def __init__(self):
        self.introduction = ""
        self.includesParticular = []
        self.excludesPerticular = []

    def addInclude(self, str):
        self.includesParticular.append(str)

    def addExclude(self, str):
        self.excludesPerticular.append(str)


pattern = r"(\(<ClassLink [^>]*?/>\))"
file_path = Path("data/ncl-20240101-en-classification_texts-20230606.xml")
content = file_path.read_bytes()
decoded_content = content.decode("utf-8")
result = re.sub(pattern, "", decoded_content)
result_bytes = result.encode("utf-8")

tree = etree.parse("data/ncl-20240101-classification_top_structure-20230606.xml")
root = tree.getroot()

textTree = etree.parse(BytesIO(result_bytes))
textRoot = textTree.getroot()
textNamespace = {"ns": "http://www.wipo.int/classifications/ncl"}


def findClassTexts(classId):
    result = ClassText()
    child = textRoot.xpath(
        f'//ns:ClassTexts[@idRef="{classId}"]', namespaces=textNamespace
    )[0]
    result.id = child.attrib["id"]
    for headingItem in child.xpath(f".//ns:HeadingItem", namespaces=textNamespace):
        result.addHeadingItem(headingItem.text.strip())
    result.explanatoryNote.introduction = child.xpath(
        f".//ns:ExplanatoryNote/ns:Introduction", namespaces=textNamespace
    )[0].text.strip()
    for includeItem in child.xpath(
        f".//ns:ExplanatoryNote/ns:IncludesInParticular/ns:Include",
        namespaces=textNamespace,
    ):
        result.explanatoryNote.addInclude(includeItem.text.strip())
    for excludeItem in child.xpath(
        f".//ns:ExplanatoryNote/ns:ExcludesInParticular/ns:Exclude",
        namespaces=textNamespace,
    ):
        result.explanatoryNote.addExclude(excludeItem.text.strip())
    return result


def findGoodOrServiceTexts(goodOrServiceId):
    child = textRoot.xpath(
        f'//ns:GoodOrServiceTexts[@idRef="{goodOrServiceId}"]', namespaces=textNamespace
    )[0]
    result = []
    for label in child.xpath(f".//ns:Label", namespaces=textNamespace):
        result.append(label.text.strip())
    return result


classes = []
for child in root:
    niceclass = NiceClass()
    niceclass.id = child.attrib["id"]
    niceclass.isGoodOrService = child.attrib["isGoodOrService"]
    niceclass.classNumber = int(child.attrib["classNumber"])
    niceclass.classTexts = findClassTexts(niceclass.id)
    for goodNode in child.xpath(f".//ns:GoodOrService", namespaces=textNamespace):
        goods = GoodOrService()
        goods.basicNumber = goodNode.attrib["basicNumber"]
        goods.id = goodNode.attrib["id"]
        goods.labels = findGoodOrServiceTexts(goods.id)
        niceclass.goodOrServices.append(goods)
    classes.append(niceclass)

dicts = []
for niceclass in classes:
    dicts.append(niceclass.toDict())

df = pd.DataFrame(dicts)
df.to_json("data/output.json", orient="records", indent=4, force_ascii=False)
df.to_pickle("data/output.pkl")
df.to_csv("data/output.csv", index=False)
