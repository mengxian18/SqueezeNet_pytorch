import os.path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

transform_BZ = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)


class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
            transforms.Resize(224),  # 调整图像大小为224x224
            transforms.RandomHorizontalFlip(),  # 随机左右翻转图像
            transforms.RandomVerticalFlip(),  # 随机上下翻转图像
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
            transform_BZ  # 执行某些复杂变换操作
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize(224),  # 调整图像大小为224x224
            transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
            transform_BZ  # 执行某些复杂变换操作
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split(' '), imgs_info))
        return imgs_info

    def padding_black(self, img):
        w, h = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]

        img_path = os.path.join('', img_path)
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs_info)