import os
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, yolo_file):
    in_file = open(xml_file)
    out_file = open(yolo_file, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text.lower().replace(' ', '_')  # Sınıf ismini küçük harfe çevir ve boşlukları alt çizgi ile değiştir
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

if __name__ == "__main__":
    classes = ["with_helmet", "without_helmet"]
    input_dir = "/Users/efenalbant1/helmet_detection/helmet-env/data/annotations"  # XML dosyalarının bulunduğu klasör
    output_dir = "/Users/efenalbant1/helmet_detection/yolov5/data/dataset"  # YOLO formatındaki etiket dosyalarının kaydedileceği klasör

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(input_dir):
        if xml_file.endswith(".xml"):
            yolo_file = os.path.join(output_dir, os.path.splitext(xml_file)[0] + ".txt")
            convert_annotation(os.path.join(input_dir, xml_file), yolo_file)