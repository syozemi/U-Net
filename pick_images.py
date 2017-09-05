import pickle
import yaml

f = open("../U-Net_Gsan/settings.yml", encoding='UTF-8')
settings = yaml.load(f)
print(settings)

sample_original = [[1,2,3], [3,4,5]]
sample_rotate = [[3,2,5], [8,4,5]]
# ランダムにn割取り出す
def pick_images(xt_ratio, effective_image):
    return image_x, image_t

def load_images(effective_image):
    images_x = []
    images_t = []
    for type, value in effective_image:
        if value == "valid":
            with open('data/image_{0}'.format(type), 'rb') as f:
                images_x.append(pickle.load(f))
            with open('data/image_answer_{0}'.format(type), 'rb') as f:
                images_t.append(pickle.load(f))
    return images_x, images_t
