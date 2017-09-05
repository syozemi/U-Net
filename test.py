import pickle

with open('data/image572', 'rb') as f:
    image_x = pickle.load(f)

with open('data/nucleus_label', 'rb') as f:
    image_t = pickle.load(f)

# print("image_x")
# print(image_x)
# print("image_t")
# print(image_t)
print(len(image_t))
print(len(image_x))
