def visualize_sample(image, target):
    for box, label in zip(target['boxes'], target['labels']):
        label = CLASSES[label]
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
        )
        cv2.putText(
            image, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

