import os
import re
import csv

from pycocotools.coco import COCO

class COCOInterpreter:


    def __init__(self, root_dir='', map_file='', protocol='train'):
        """
        Args:
            root_dir: root directory to MSCOCO dataset
            protocol: 'train' or 'val'
        """
        print(root_dir)
        assert(os.path.isdir(root_dir))
        anno_dir = os.path.join(root_dir, 'annotations')

        # annotation file of training and validation set
        det_anno_file = 'instances_%s2014.json'
        det_anno_file = os.path.join(anno_dir, det_anno_file % (protocol))
        cap_anno_file = 'captions_%s2014.json'
        cap_anno_file = os.path.join(anno_dir, cap_anno_file % (protocol))

        self.coco_det = COCO(det_anno_file)
        self.coco_cap = COCO(cap_anno_file)
        self.root_dir = root_dir
        self.map_file = map_file
        self.det2cap = None
        self.protocol = protocol

    def get_det2cap(self):
        """Retrieve mapping of caption labels and detection labels."""
        # retrieve manual mapping of 1000 caption labels to 73 detection labels
        # mapping_file = open('dataset/mscoco_vc/coco2vocab_manual_mapping.txt', 'r')
        assert(os.path.isfile(self.map_file))
        mapping_file = open(self.map_file, 'r')
        mapping = mapping_file.read()
        # split by "\n"
        mapping = re.split('\n', mapping)[:-1]
        # mapping is formatted as 'a: a1, a2, a3\r', then we split it by ':,\r'
        mapping_dict = {}
        for vocab_map in mapping:
            vocabs = re.split('[:,\r]', vocab_map)

            vocabs = [item for item in vocabs if item not in ('', ' ')]
            if len(vocabs) > 1:
                caption_vocabs = [i.replace(' ','') for i in vocabs[1:]]
                det_id = self.coco_det.getCatIds(vocabs[0])[0]
                mapping_dict.setdefault(det_id, caption_vocabs)

        return mapping_dict

    def get_labels(self):
        """Retrive visual labels and visual concepts of each image in the
        dataset.
        """
        if not self.det2cap:
            self.det2cap = self.get_det2cap()
        det2cap = self.det2cap
        det_ids = det2cap.keys()
        det_id_indices = {det_id: ind for ind, det_id in enumerate(det_ids)}

        # reverse the det2cap, generate mapping of caption labels to detection
        # category id
        cap2det = {cap: det for det, caps in det2cap.iteritems() for cap in caps}

        detset = self.coco_det
        capset = self.coco_cap
        protocol = self.protocol
        img_dir = os.path.join(self.root_dir, 'images', protocol+'2014')

        # retrieve images with detection bounding boxes
        img_ids = detset.getImgIds()
        results = {}
        for img_id in img_ids:
            # retrieve detection labels
            det_ann_ids = detset.getAnnIds(imgIds=img_id, catIds=det_ids)
            if not det_ann_ids:
                continue
            img_name = detset.loadImgs(img_id)[0]['file_name']
            det_ids_in_img = [ann['category_id'] for ann in
                              detset.loadAnns(det_ann_ids)]

            # format visual labels from detection labels as MIL detection
            visual_labels = [0] * len(det_ids)
            visual_label_cnt = [0] * len(det_ids)
            for det_id in det_ids_in_img:
                ind = det_id_indices[det_id]
                visual_labels[ind] = 1
                visual_label_cnt[ind] += 1

            # retrieve caption labels
            cap_ann_ids = capset.getAnnIds(imgIds=img_id)
            caps_ = capset.loadAnns(cap_ann_ids)
            caps = [item['caption'].encode("utf-8").lower() for item in caps_]

            # format visual concepts from captions labels as MIL detection
            # split captions by ' '
            visual_concepts = [0] * len(det_ids)
            visual_concept_cnt = [0] * len(det_ids)
            for cap in caps:
                rm_dot_cap = cap.replace('.', '')
                vocabs = rm_dot_cap.split(' ')
                vocab_ids = [cap2det.get(voc, None) for voc in vocabs]
                vocab_ids = [vid for vid in vocab_ids if vid]
                for vid in vocab_ids:
                    ind = det_id_indices[vid]
                    # skip if the concept not belongs to visual label
                    if visual_labels[ind] > 0:
                        visual_concepts[ind] = 1
                        visual_concept_cnt[ind] += 1

            full_img_name = os.path.join(self.root_dir, img_name)
            labels_concepts = {'visual_labels': visual_labels,
                               'visual_label_cnt': visual_label_cnt,
                               'visual_concepts': visual_concepts,
                               'visual_concept_cnt': visual_concept_cnt}
            results.setdefault(full_img_name, labels_concepts)

        return results

    def write2csv(self, results, csv_path):
        """Write results to csv.
        """
        results = self.get_labels()

        assert(csv_path)
        with open(csv_path, 'wb') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for img_name, label_dict in results.iteritems():
                csv_str = img_name.encode('utf-8')
                sample_keys = ['visual_labels',
                               'visual_concepts',
                               'visual_label_cnt',
                               'visual_concept_cnt']
                for key in sample_keys:
                    value = label_dict[key]
                    str_value = ''.join(str(i) for i in value)
                    csv_str = csv_str + str_value
                writer.writerow(csv_str)
