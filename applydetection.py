import cv2

image = cv2.imread("test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (x, y, window) in sliding_window(gray, step_size=16, window_size=(128, 128)):
    if window.shape[0] != 128 or window.shape[1] != 128:
        continue

    features = hog(
        window,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    prediction = model.predict([features])

    if prediction == 1:
        cv2.rectangle(image, (x, y),
                       (x + 128, y + 128),
                       (0, 255, 0), 2)

cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
