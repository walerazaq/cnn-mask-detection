class maskSet(Dataset):
    def __init__(self, dataset, width, height, classes, img_dir_path, xml_dir_path, transforms=None):
        self.transforms = transforms
        self.img_dir_path = img_dir_path
        self.xml_dir_path = xml_dir_path
        self.height = height
        self.width = width
        self.dataset = dataset
        self.classes = classes
        
    def __getitem__(self, idx):
        
        image_name = self.dataset[idx]
        image_path = os.path.join(self.img_dir_path, image_name)
        
        image = cv2.imread(image_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.xml_dir_path, annot_filename)
             
        boxes = []
        labels = []

        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        for i in root.findall('size'):
            image_width = int(i.find('width').text)
            image_height = int(i.find('height').text)
        
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height
            
            if xmin_final > self.width:
                xmin_final = self.width

            if ymin_final > self.height:
                ymin_final = self.height

            if xmax_final > self.width:
                xmax_final = self.width

            if ymax_final > self.height:
                ymax_final = self.height
                
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        if boxes.shape[0] > 1:
            iscrowd = torch.ones((boxes.shape[0],), dtype=torch.int64)
        else:
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']

            target["boxes"] = torch.Tensor(sample['bboxes'])
        
            return image_resized, target
        
        else:
            image_resized = np.transpose(image_resized, (2, 0, 1))
            image_resized = torch.from_numpy(image_resized)
            
            return image_resized, target
    
    def __len__(self):
        return len(self.dataset)
