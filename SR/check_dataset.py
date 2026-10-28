from PIL import Image
import matplotlib.pyplot as plt

input_path = "dataset/train/input/0001.png"
target_path = "dataset/train/target/0001.png"

input_image = Image.open(input_path)
target_image = Image.open(target_path)

print("Input size :", input_image.size)
print("Target size:", target_image.size)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Degraded Input")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(target_image)
plt.title("HR Target")
plt.axis("off")

plt.tight_layout()

plt.savefig("dataset_pair_check.png")

plt.show()