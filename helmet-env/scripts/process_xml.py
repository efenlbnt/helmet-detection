import os
import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_df(xml_folder):
    xml_list = []
    for xml_file in os.listdir(xml_folder):
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (root.find('filename').text,
                     int(root.find('size').find('width').text),
                     int(root.find('size').find('height').text),
                     member.find('name').text,
                     int(bndbox.find('xmin').text),
                     int(bndbox.find('ymin').text),
                     int(bndbox.find('xmax').text),
                     int(bndbox.find('ymax').text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# XML dosyalarının bulunduğu klasör yolu
annotations_path = 'helmet-env/data/annotations'

# DataFrame'e dönüştür
annotations_df = xml_to_df(annotations_path)

# DataFrame'i CSV dosyası olarak kaydet
annotations_df.to_csv('/Users/efenalbant1/helmet_detection/helmet-env/data/annotations.csv', index=False)

# İlk beş etiketi göster
print(annotations_df)