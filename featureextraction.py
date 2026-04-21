def extract_hog(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

X = []
y = []

# Positive images
for img in os.listdir("data/positive"):
    X.append(extract_hog(f"data/positive/{img}"))
    y.append(1)

# Negative images
for img in os.listdir("data/negative"):
    X.append(extract_hog(f"data/negative/{img}"))
    y.append(0)

X = np.array(X)
y = np.array(y)
